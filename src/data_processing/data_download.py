# !pip install requests
import glob
import ast
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import shutil
import zipfile
import pandas as pd
import numpy as np
import json
import os
import requests

# Define the URL and file paths

def download_zip_files(urls, raw_dir):
    zips_path = raw_dir

    # Create directories
    os.makedirs(zips_path, exist_ok=True)

    # Download each zip file
    for url in urls:
        file_name = url.split("/")[-1]
        zip_path = os.path.join(zips_path, file_name)
        
        response = requests.get(url)
        with open(zip_path, 'wb') as file:
            file.write(response.content)
        
        print(f"Download complete. File stored as {file_name}.")



#TILL HERE, DOWNLOADING THE ZIP FILES AND STORING THEM
###########################################################
#FROM HERE, MAKING THE PLAYER WISE CSV FILES FOR ALL MATCHES



#How to use:
# Make a new directory where you want to store the csv Files 
# Download the Json files in the directory
# On your desktop copy the path of the directory that you had earlier created
# and paste it in the fields of os.chdir() on the lines 153 and 226
# Run the Code
# The final ouput csv files will be in the folder Interim Excel Files
    
def extract_and_convert_to_csv(raw_dir, interim_dir):
    pd.set_option('display.max_columns', None)


    def match_info_extract(data):
        match_info = {
            "date_of_the_match": data["info"]["dates"][0],
            "match_type": data["info"]["match_type"],
            "venue": data["info"]["venue"],
            "event": data["info"].get("event", np.nan),
            "season": data["info"]["season"],
            "toss_winner": data["info"]["toss"]["winner"],
            "toss_decision": data["info"]["toss"]["decision"],
            "winner": data["info"]["outcome"].get("winner", np.nan),
            # "winner": data["info"]["outcome"]["winner"], # If the match is drawn, this will be 'draw'
            "player_of_match": data["info"].get("player_of_match", np.nan),
            "teams": data["info"]["teams"],
            "city": data["info"].get("city", np.nan),
            "match_number": data["info"].get("match_number", np.nan)
        }
        if (isinstance(match_info["player_of_match"], list)):
            match_info["player_of_match"] = match_info["player_of_match"][0]

        return match_info


    # Extracting innings information

    def make_innings_table(data):
        innings_list = []
        for inning_no, inning in enumerate(data['innings'], start=1):
            team = inning['team']
            for over in inning.get('overs', []):
                over_number = over['over']
                for delivery in over.get('deliveries', []):
                    # Extract fielder names for any wickets
                    fielder_names = [
                        fielder.get('name')
                        for wicket in delivery.get('wickets', [])
                        if wicket.get('fielders')
                        for fielder in wicket.get('fielders', [])
                    ] or [None]  # Default to None if no fielder
                    # print(type(inning_no))
                    delivery_info = {
                        'inning_no': int(int(inning_no/2)+inning_no % 2),
                        'team': team,
                        'over': over_number,
                        'ball_number': delivery.get('ball'),
                        'batter': delivery.get('batter', None),
                        'bowler': delivery.get('bowler', None),
                        'non_striker': delivery.get('non_striker', None),
                        'runs_batter': delivery.get('runs', {}).get('batter', 0),
                        'runs_extras': delivery.get('runs', {}).get('extras', 0),
                        'runs_total': delivery.get('runs', {}).get('total', 0),
                        'wides': delivery.get('extras', {}).get('wides', 0),
                        'noballs': delivery.get('extras', {}).get('noballs', 0),
                        'byes': delivery.get('extras', {}).get('byes', 0),
                        'legbyes': delivery.get('extras', {}).get('legbyes', 0),
                        'penalty': delivery.get('extras', {}).get('penalty', 0),
                        'wicket': delivery.get('wickets', [{}])[0].get('kind', None),
                        'player_out': delivery.get('wickets', [{}])[0].get('player_out', None),
                        'fielder_names': fielder_names
                    }
                    innings_list.append(delivery_info)

        innings_df = pd.DataFrame(innings_list)
        return innings_df


    def make_player_table(data, match_info, innings_df, filename):
        players_data = []
        match_type = match_info["match_type"]

        if match_type == 'Test' or match_type == 'MDM':
            # Process each inning separately
            # print("aaaaaaaaaaaa")
            for inning_no in innings_df['inning_no'].unique():

                inning_df = innings_df[innings_df['inning_no'] == inning_no]
                # print(inning_df)
                team = inning_df['team'].iloc[0]
                opponent_team = [t for t in match_info['teams'] if t != team][0]
                players = data["info"]["players"][team]
                players1 = data["info"]["players"][opponent_team]
                # print(players1)
                players = players + players1
                # print(inning_df)

                for player in players:
                    # Calculate player statistics for this inning
                    catches_taken = inning_df[
                        (inning_df['fielder_names'].apply(lambda x: player in x if x else False)) &
                        (inning_df['wicket'].notna()) &
                        (inning_df['wicket'] == 'caught')
                    ].shape[0]
                    if player in players1:
                        team, opponent_team = opponent_team, team
                    # print(player)
                    # print(players1)
                    # print(player in players1)
                    player_data = {
                        "date_of_the_match": match_info["date_of_the_match"],
                        "match_type": match_info["match_type"],
                        "inning_no": inning_no,
                        "venue": match_info["venue"],
                        "event": match_info["event"],
                        "season": match_info["season"],
                        "toss_winner": match_info["toss_winner"],
                        "toss_decision": match_info["toss_decision"],
                        "winner": match_info["winner"],
                        "player_name": player,
                        "player_id": data["info"]["registry"]["people"].get(player, np.nan),
                        "team_name": team,
                        "opponent_team": opponent_team,
                        "runs_scored": inning_df[inning_df['batter'] == player]['runs_batter'].sum(),
                        "balls_played": inning_df[inning_df['batter'] == player].shape[0],
                        "balls_thrown": inning_df[inning_df['bowler'] == player].shape[0],
                        # "wickets_taken": inning_df[inning_df['bowler'] == player]['wicket'].notna().sum(),
                        "extras_given": inning_df[inning_df['bowler'] == player][['wides', 'noballs', 'byes', 'legbyes', 'penalty']].sum().sum(),
                        "fours": inning_df[(inning_df['batter'] == player) & (inning_df['runs_batter'] == 4)].shape[0],
                        "sixes": inning_df[(inning_df['batter'] == player) & (inning_df['runs_batter'] == 6)].shape[0],
                        "catches_taken": catches_taken,  # Updated with calculated catches count
                        # "overs_bowled": inning_df[inning_df['bowler'] == player]['over'].nunique(),
                        "match_number": data["info"].get("event", {}).get("match_number", np.nan),
                        "city": data["info"].get("city", np.nan),
                        "filename": filename,
                        "player_of_the_match_yes_or_no": 1 if player == match_info["player_of_match"] else 0,
                        "out_kind": inning_df[inning_df['player_out'] == player]['wicket'].values[0] if inning_df[inning_df['player_out'] == player].shape[0] > 0 else np.nan,
                        "runs_given": inning_df[inning_df['bowler'] == player]['runs_total'].sum(),
                        "wickets_taken": inning_df[inning_df['bowler'] == player]['wicket'].notna().sum(),
                        "fours_given": inning_df[(inning_df['bowler'] == player) & (inning_df['runs_batter'] == 4)].shape[0],
                        "sixes_given": inning_df[(inning_df['bowler'] == player) & (inning_df['runs_batter'] == 6)].shape[0]
                        # Add more stats if needed

                    }
                    player_data["overs_bowled"] = 0
                    player_data["dot_balls"] = 0
                    init = -25
                    player_data["wickettypes"] = {
                        "bowled": 0, "lbw": 0, "caught": 0, 'stumped': 0, 'run out': 0, 'caught and bowled': 0}
                    runs = 0
                    bowled = 0
                    player_data["maiden_overs"] = 0
                    df = inning_df[inning_df['bowler'] == player]
                    for i in df.index:
                        runs += df.at[i, 'runs_total']
                        if (df.at[i, 'runs_total']):
                            player_data["dot_balls"] += 1
                        if (i-init > 1):
                            player_data["overs_bowled"] += 1
                            if (runs == 0):
                                player_data["maiden_overs"] += 1
                            runs = 0
                        init = i
                        if (df.at[i, 'wicket']):
                            if (df.at[i, 'wicket'] in player_data["wickettypes"]):
                                player_data["wickettypes"][df.at[i, 'wicket']] += 1
                            else:
                                player_data["wickettypes"][df.at[i, 'wicket']] = 0
                    t1 = inning_df[inning_df['batter'] == player]
                    if (not t1.empty):
                        player_data["over_faced_first"] = t1.loc[t1.index[0], "over"]
                    for i in t1.index:
                        if (t1.at[i, 'wicket']):
                            player_data["out_by_bowler"] = t1.loc[i, "bowler"]
                            player_data["out_by_fielder"] = t1.loc[i,
                                                                "fielder_names"]
                    players_data.append(player_data)
                    if player in players1:
                        team, opponent_team = opponent_team, team
        else:
            # Process the match normally for non-Test matches
            for team in match_info["teams"]:
                opponent_team = [t for t in match_info['teams'] if t != team][0]
                players = data["info"]["players"][team]

                for player in players:
                    catches_taken = innings_df[
                        (innings_df['fielder_names'].apply(lambda x: player in x if x else False)) &
                        (innings_df['wicket'].notna()) &
                        (innings_df['wicket'] == 'caught')
                    ].shape[0]

                    player_data = {
                        "date_of_the_match": match_info["date_of_the_match"],
                        "match_type": match_info["match_type"],
                        "venue": match_info["venue"],
                        "event": match_info["event"],
                        "season": match_info["season"],
                        "toss_winner": match_info["toss_winner"],
                        "toss_decision": match_info["toss_decision"],
                        "winner": match_info["winner"],
                        "player_name": player,
                        "player_id": data["info"]["registry"]["people"].get(player, np.nan),
                        "team_name": team,
                        "opponent_team": opponent_team,
                        "runs_scored": innings_df[innings_df['batter'] == player]['runs_batter'].sum(),
                        "balls_played": innings_df[innings_df['batter'] == player].shape[0],
                        "overs_bowled": innings_df[innings_df['bowler'] == player]['over'].nunique(),
                        # Add more stats if needed
                        "balls_thrown": innings_df[innings_df['bowler'] == player].shape[0],
                        # "wickets_taken": innings_df[innings_df['bowler'] == player]['wicket'].notna().sum(),
                        "extras_given": innings_df[innings_df['bowler'] == player][['wides', 'noballs', 'byes', 'legbyes', 'penalty']].sum().sum(),
                        "fours": innings_df[(innings_df['batter'] == player) & (innings_df['runs_batter'] == 4)].shape[0],
                        "sixes": innings_df[(innings_df['batter'] == player) & (innings_df['runs_batter'] == 6)].shape[0],
                        "catches_taken": catches_taken,  # Updated with calculated catches count
                        # "overs_bowled": innings_df[innings_df['bowler'] == player]['over'].nunique(),
                        "match_number": data["info"].get("event", {}).get("match_number", np.nan),
                        "city": data["info"].get("city", np.nan),
                        "filename": filename,
                        "player_of_the_match_yes_or_no": 1 if player == match_info["player_of_match"] else 0,
                        "out_kind": innings_df[innings_df['player_out'] == player]['wicket'].values[0] if innings_df[innings_df['player_out'] == player].shape[0] > 0 else np.nan,
                        "runs_given": innings_df[innings_df['bowler'] == player]['runs_total'].sum(),
                        "wickets_taken": innings_df[innings_df['bowler'] == player]['wicket'].notna().sum(),
                        "fours_given": innings_df[(innings_df['bowler'] == player) & (innings_df['runs_batter'] == 4)].shape[0],
                        "sixes_given": innings_df[(innings_df['bowler'] == player) & (innings_df['runs_batter'] == 6)].shape[0]

                    }
                    player_data["overs_bowled"] = 0
                    player_data["dot_balls"] = 0
                    init = -25
                    player_data["wickettypes"] = {
                        "bowled": 0, "lbw": 0, "caught": 0, 'stumped': 0, 'run out': 0, 'caught and bowled': 0}
                    runs = 0
                    bowled = 0
                    player_data["maiden_overs"] = 0
                    df = innings_df[innings_df['bowler'] == player]
                    for i in df.index:
                        runs += df.at[i, 'runs_total']
                        if (df.at[i, 'runs_total']):
                            player_data["dot_balls"] += 1
                        if (i-init > 1):
                            player_data["overs_bowled"] += 1
                            if (runs == 0):
                                player_data["maiden_overs"] += 1
                            runs = 0
                        init = i
                        if (df.at[i, 'wicket']):
                            if (df.at[i, 'wicket'] in player_data["wickettypes"]):
                                player_data["wickettypes"][df.at[i, 'wicket']] += 1
                            else:
                                player_data["wickettypes"][df.at[i, 'wicket']] = 0
                    t1 = innings_df[innings_df['batter'] == player]
                    if (not t1.empty):
                        player_data["over_faced_first"] = t1.loc[t1.index[0], "over"]
                    for i in t1.index:
                        if (t1.at[i, 'wicket']):
                            player_data["out_by_bowler"] = t1.loc[i, "bowler"]
                            player_data["out_by_fielder"] = t1.loc[i,
                                                                "fielder_names"]
                    # players_data.append(player_data)

                    players_data.append(player_data)

        players_df = pd.DataFrame(players_data)

        return players_df


    def download_data(innings_df, name, file_name):
        folder_path = "./"+file_name+"_csv"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        os.chdir(folder_path)
        innings_df.to_csv(name + '.csv', index=False)
        os.chdir("../")

    # Loading JSON Data


    # Loading the JSON data from the file
    def to_csv(file_name):
        # Replace 'your_folder.zip' with the name of your uploaded file
        os.chdir(raw_dir)
        print(os.getcwd())
        zip_file_name = "./"+file_name+"_json.zip"
        extract_dir = file_name + "s"  # specify the directory to extract to

        # Unzip the folder
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        for root, dirs, files in os.walk(extract_dir):
            for x in files:
                with open(os.path.join(root, x), 'r') as file:
                    try:
                        data = json.load(file)
                        print(file)
                        # match_df = make_match_table(data)
                        match_info = match_info_extract(data)
                        innings_df = make_innings_table(data)
                        player_df = make_player_table(
                            data, match_info, innings_df, x)

                        # players_df = make_player_table(data
                        # ball_by_ball_df = make_ball_by_ball_table(data)
                        download_data(player_df, x[:-5], file_name)
                    except Exception as e:
                        print("For file: ", file)
                        print(f"An error occurred: {e}")
                        continue
                    # data = json.load(file)
                    # print(file)
                    # # match_df = make_match_table(data)
                    # match_info = match_info_extract(data)
                    # innings_df = make_innings_table(data)
                    # player_df = make_player_table(data, match_info, innings_df, x)
                    # download_data(player_df, x[:-5],file_name)

    # Display the loaded data


    def concate(file_name):
        print(os.getcwd())
        folder_path = "./" + file_name + "_csv/"
        file_no = 0
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        df_list = list()
        print(len(csv_files))
        concatenated_df = pd.DataFrame()
        total_files = len(csv_files)
        for filename in csv_files:
            file_path = filename
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                df_list.append(df)
                file_no += 1
                print("For the league: " + file_name)
                print("Files Processed : " + str(file_no) + " / ", total_files)
                print("Percentage Done ", (file_no/total_files)*100)
            concatenated_df = pd.concat(df_list, ignore_index=True)
        output_path = "./Interim Excel Files"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        os.chdir(output_path)
        output_file = file_name+".csv"
        concatenated_df.to_csv(output_file, index=False)
        os.chdir("../")


    def find_zip_files(directory):
        # Look for .zip files in the specified directory
        zip_files = glob.glob(f"{directory}/*.zip")
        return zip_files


    def delete_folders_except(directory, exception):
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path) and folder != exception:
                shutil.rmtree(folder_path)


    # Usage example
    # def main():
    os.chdir(raw_dir)
    zip_files = find_zip_files("./")
    print(zip_files)
    names = list()
    for i in zip_files:
        rt = i[:-9]
        lt = rt[2:]
        names.append(lt)
    print("check2")
    print(names)
    for i in names:
        print("Processing for the league: " + i)
        to_csv(i)
        print("check1")
        concate(i)
    delete_folders_except(
        raw_dir, "Interim Excel Files")
    # Move the Interim Excel Files folder to the Github Files directory
    
    temp_dir=raw_dir+"/Interim Excel Files"
    source_dir = temp_dir
    destination_dir =interim_dir
    if os.path.exists(source_dir):
        shutil.move(source_dir, destination_dir)
    #   process_all_files()













def process_final_csv(interim_dir, processed_dir):
    pts = {"Test": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 0,
        "half_cen": 4,
        "century": 4,
        "duck": -4,
        "wicket": {"lbw": 24, "bowled": 24, "caught": 16, "stumped": 12, "run out": 12, "catches_taken": 8},
        "hauls": {0: 0, 1: 0, 2: 0, 3: 0, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8},
        "maiden": 0,
        "minovers": 10000,
        "minballs": 100000000
    }, "MDM": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 0,
        "half_cen": 4,
        "century": 4,
        "duck": -4,
        "wicket": {"lbw": 24, "bowled": 24, "caught": 16, "stumped": 12, "run out": 12, "catches_taken": 8},
        "hauls": {0: 0, 1: 0, 2: 0, 3: 0, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8},
        "maiden": 0,
        "minovers": 10000,
        "minballs": 100000000
    }, "ODI": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 0,
        "half_cen": 4,
        "century": 4,
        "duck": -3,
        "wicket": {"lbw": 33, "bowled": 33, "caught": 25, "stumped": 12, "run out": 12, "catches_taken": 8, "3catches_taken": 4},
        "hauls": {0: 0, 1: 0, 2: 0, 3: 0, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8},
        "maiden": 4,
        "minovers": 5,
        "economy": {2.5: 6, 3.5: 4, 4.5: 2, 7: 0, 8.01: -2, 9.01: -4, 100: -6},
        "minballs": 20,
        "strike_rate": {30: -6, 40: -4, 50: -2, 100: 0, 120.01: 2, 140.01: 4, 1000: 6}
    }, "ODM": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 0,
        "half_cen": 4,
        "century": 4,
        "duck": -3,
        "wicket": {"lbw": 33, "bowled": 33, "caught": 25, "stumped": 12, "run out": 12, "catches_taken": 8, "3catches_taken": 4},
        "hauls": {0: 0, 1: 0, 2: 0, 3: 0, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8},
        "maiden": 4,
        "minovers": 5,
        "economy": {2.5: 6, 3.5: 4, 4.5: 2, 7: 0, 8.01: -2, 9.01: -4, 100: -6},
        "minballs": 20,
        "strike_rate": {30: -6, 40: -4, 50: -2, 100: 0, 120.01: 2, 140.01: 4, 1000: 6},
    },  "T20": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 4,
        "half_cen": 4,
        "century": 8,  # these points are cumulative
        "duck": -2,
        "wicket": {"lbw": 33, "bowled": 33, "caught": 25, "stumped": 12, "run out": 12, "catches_taken": 8, "3catches_taken": 4},
        "hauls": {0: 0, 1: 0, 2: 0, 3: 4, 4: 8, 5: 16, 6: 16, 7: 16, 8: 16, 9: 16, 10: 16},
        "maiden": 12,
        "minovers": 2,
        "economy": {5: 6, 6: 4, 7: 2, 10: 0, 11.01: -2, 12.01: -4, 100: -6},
        "minballs": 10,
        "strike_rate": {50: -6, 60: -4, 70: -2, 130: 0, 150.01: 2, 170.01: 4, 1000: 6}
    },  "IT20": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 4,
        "half_cen": 4,
        "century": 8,  # these points are cumulative
        "duck": -2,
        "wicket": {"lbw": 33, "bowled": 33, "caught": 25, "stumped": 12, "run out": 12, "catches_taken": 8, "3catches_taken": 4},
        "hauls": {0: 0, 1: 0, 2: 0, 3: 4, 4: 8, 5: 16, 6: 16, 7: 16, 8: 16, 9: 16, 10: 16},
        "maiden": 12,
        "minovers": 2,
        "economy": {5: 6, 6: 4, 7: 2, 10: 0, 11.01: -2, 12.01: -4, 100: -6},
        "minballs": 10,
        "strike_rate": {50: -6, 60: -4, 70: -2, 130: 0, 150.01: 2, 170.01: 4, 1000: 6}
    },
        "T10": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 8,
        "half_cen": 8,
        "century": 0,  # these points are cumulative
        "duck": -2,
        "wicket": {"lbw": 33, "bowled": 33, "caught": 25, "stumped": 12, "run out": 12, "catches_taken": 8, "3catches_taken": 4},
        "hauls": {0: 0, 1: 0, 2: 8, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 16, 9: 16, 10: 16},
        "maiden": 16,
        "minovers": 1,
        "economy": {7: 6, 8: 4, 9: 2, 14: 0, 15.01: -2, 16.01: -4, 100: -6},
        "minballs": 5,
        "strike_rate": {60: -6, 70: -4, 80.01: -2, 150.01: 0, 170.01: 2, 190.01: 4, 1000: 6}
    },
        "6ixty": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 8,
        "half_cen": 8,
        "century": 0,  # these points are cumulative
        "duck": -2,
        "wicket": {"lbw": 33, "bowled": 33, "caught": 25, "stumped": 12, "run out": 12, "catches_taken": 8, "3catches_taken": 4},
        "hauls": {0: 0, 1: 0, 2: 8, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 16, 9: 16, 10: 16},
        "maiden": 16,
        "minovers": 1,
        "economy": {7: 6, 8: 4, 9: 2, 14: 0, 15.01: -2, 16.01: -4, 100: -6},
        "minballs": 5,
        "strike_rate": {60: -6, 70: -4, 80.01: -2, 150.01: 0, 170.01: 2, 190.01: 4, 1000: 6}
    },
        "Hundred": {
        "runs": 1,
        "boundary": 1,
        "sixes": 2,
        "30run": 5,
        "half_cen": 5,
        "century": 10,  # these points are cumulative
        "duck": -2,
        "wicket": {"lbw": 33, "bowled": 33, "caught": 25, "stumped": 12, "run out": 12, "catches_taken": 8, "3catches_taken": 4},
        "hauls": {0: 0, 1: 0, 2: 3, 3: 5, 4: 10, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20},
        "maiden": 0,
        "minovers": 100000,
        "economy": {7: 6, 8: 4, 9: 2, 14: 0, 15.01: -2, 16.01: -4, 100: -6},
        "minballs": 10000000,
        "strike_rate": {60: -6, 70: -4, 80.01: -2, 150.01: 0, 170.01: 2, 190.01: 4, 1000: 6}
    }
    }


    # wickets = {type_wicket: number} #role="playing11"
    def calculate_fantasy_points(match_type, player_type, role, runs_scored, fours, sixes, wickets, multiplier, overs_bowled, balls_played, runsgiven, maidens):
        # match_type can be Test,T20,ODI etc.
        strike_rate = 0
        if balls_played > 0:
            strike_rate = runs_scored/balls_played*100
        economy = 0
        if overs_bowled > 0:
            economy = runsgiven/overs_bowled

        other_type = 0
        if (match_type[:5] == "Other"):
            other_type = 1
            match_type = match_type
        # print(economy)
        # match_type="ODI"
        # Implement the scoring calculation here

        batting_points = pts[match_type]["runs"]*runs_scored  # adding runs_scored
        # addings boundaries and sixes
        batting_points += (fours*pts[match_type]
                        ["boundary"] + sixes*pts[match_type]["sixes"])
        if balls_played > 0:
            batting_points += ((runs_scored == 0 & (player_type != "bowler"))
                            * pts[match_type]["duck"])  # points for duck
        batting_points += ((runs_scored >= 30)*pts[match_type]["30run"] + (runs_scored >= 50)*pts[match_type]["half_cen"]+(
            # adding bonuses for 30,50,100
            runs_scored >= 100)*pts[match_type]["century"])
        # print(pts[match_type]["economy"])
        # print(batting_points)
        if ((overs_bowled >= pts[match_type]["minovers"])):
            for x, y in pts[match_type]["economy"].items():
                if (economy < x):
                    economy_multiplier = y
                    break
        else:
            economy_multiplier = 0
        if ((balls_played >= pts[match_type]["minballs"])):
            for x, y in pts[match_type]["strike_rate"].items():
                if (strike_rate < x):
                    sr_multiplier = y
                    break
        else:
            sr_multiplier = 0
        # adding sr points in batting points
        batting_points += ((balls_played >= pts[match_type]
                        ["minballs"] and player_type != "bowler")*sr_multiplier)
        # print(batting_points)
        # adding economy points in bowling points
        bowling_points = (
            (overs_bowled >= pts[match_type]["minovers"])*economy_multiplier)

        # wickets_points = {"lbw": 24, "bowled": 24, "caught": 16, "stumped": 12, "run out": 12, "catches_taken": 8, "Run_Not_Direct": 6}
        for x, y in wickets.items():
            # adding all wickets and catches_takenes types
            bowling_points += pts[match_type]["wicket"][x]*y
        # calculating the numbers of for hauls
        num_wickets = (wickets["lbw"] + wickets["bowled"] + wickets["caught"])
        # print(num_wickets)
        bowling_points += pts[match_type]["hauls"][num_wickets]  # wickethauls
        bowling_points += pts[match_type]["maiden"]*maidens  # maiden overs_bowled

        if (other_type == 0):
            # if not playing11, subs(for given reasons) no 4points
            extra_points = 4-4*(role == "others")

        points = bowling_points + batting_points + extra_points  # adding all the points
        points += (multiplier == "captain")*points + (multiplier ==
                                                    "vice_captain")*points/2  # multiplier for captain and vicecaptain

        return points
        pass


    def calculate_batting_points(match_type, player_type, runs_scored, fours, sixes, balls_played):
        strike_rate = 0
        if balls_played > 0:
            strike_rate = runs_scored / balls_played * 100

        batting_points = pts[match_type]["runs"] * runs_scored
        batting_points += (fours * pts[match_type]
                        ["boundary"] + sixes * pts[match_type]["sixes"])
        if balls_played > 0:
            batting_points += ((runs_scored == 0 and player_type !=
                            "bowler") * pts[match_type]["duck"])
        batting_points += ((runs_scored >= 30) * pts[match_type]["30run"] +
                        (runs_scored >= 50) * pts[match_type]["half_cen"] +
                        (runs_scored >= 100) * pts[match_type]["century"])

        sr_multiplier = 0
        if balls_played >= pts[match_type]["minballs"]:
            for x, y in pts[match_type]["strike_rate"].items():
                if strike_rate < x:
                    sr_multiplier = y
                    break

        batting_points += ((balls_played >= pts[match_type]["minballs"]
                        and player_type != "bowler") * sr_multiplier)

        return batting_points


    # wickets = {type_wicket: number} #role="playing11"
    def calculate_bowling_points(match_type, player_type, role, runs_scored, fours, sixes, wickets, multiplier, overs_bowled, balls_played, runsgiven, maidens):
        # match_type can be Test,T20,ODI etc.
        strike_rate = 0
        if balls_played > 0:
            strike_rate = runs_scored/balls_played*100
        economy = 0
        if overs_bowled > 0:
            economy = runsgiven/overs_bowled

        other_type = 0
        if (match_type[:5] == "Other"):
            other_type = 1
            match_type = match_type
        # print(economy)
        # match_type="ODI"
        # Implement the scoring calculation here

        batting_points = pts[match_type]["runs"]*runs_scored  # adding runs_scored
        # addings boundaries and sixes
        batting_points += (fours*pts[match_type]
                        ["boundary"] + sixes*pts[match_type]["sixes"])
        if balls_played > 0:
            batting_points += ((runs_scored == 0 & (player_type != "bowler"))
                            * pts[match_type]["duck"])  # points for duck
        batting_points += ((runs_scored >= 30)*pts[match_type]["30run"] + (runs_scored >= 50)*pts[match_type]["half_cen"]+(
            # adding bonuses for 30,50,100
            runs_scored >= 100)*pts[match_type]["century"])
        # print(pts[match_type]["economy"])
        # print(batting_points)
        if ((overs_bowled >= pts[match_type]["minovers"])):
            for x, y in pts[match_type]["economy"].items():
                if (economy < x):
                    economy_multiplier = y
                    break
        else:
            economy_multiplier = 0
        if ((balls_played >= pts[match_type]["minballs"])):
            for x, y in pts[match_type]["strike_rate"].items():
                if (strike_rate < x):
                    sr_multiplier = y
                    break
        else:
            sr_multiplier = 0
        # adding sr points in batting points
        batting_points += ((balls_played >= pts[match_type]
                        ["minballs"] and player_type != "bowler")*sr_multiplier)
        # print(batting_points)
        # adding economy points in bowling points
        bowling_points = (
            (overs_bowled >= pts[match_type]["minovers"])*economy_multiplier)

        # wickets_points = {"lbw": 24, "bowled": 24, "caught": 16, "stumped": 12, "run out": 12, "catches_taken": 8, "Run_Not_Direct": 6}
        # for x, y in wickets.items():
        #   bowling_points += pts[match_type]["wicket"][x]*y #adding all wickets and catches_takenes types
        bowling_points += pts[match_type]["wicket"]["lbw"]*wickets["lbw"]
        bowling_points += pts[match_type]["wicket"]["bowled"]*wickets["bowled"]
        bowling_points += pts[match_type]["wicket"]["caught"]*wickets["caught"]
        # calculating the numbers of for hauls
        num_wickets = (wickets["lbw"] + wickets["bowled"] + wickets["caught"])
        bowling_points += pts[match_type]["hauls"][num_wickets]  # wickethauls
        bowling_points += pts[match_type]["maiden"]*maidens  # maiden overs_bowled

        return bowling_points
        pass


    # wickets = {type_wicket: number} #role="playing11"
    def calculate_fielding_points(match_type, player_type, role, runs_scored, fours, sixes, wickets, multiplier, overs_bowled, balls_played, runsgiven, maidens):
        fielding_points = 0
        for x, y in wickets.items():
            # adding all wickets and catches_takenes types
            fielding_points += pts[match_type]["wicket"][x]*y
        fielding_points -= pts[match_type]["wicket"]["lbw"]*wickets["lbw"]
        fielding_points -= pts[match_type]["wicket"]["bowled"]*wickets["bowled"]
        fielding_points -= pts[match_type]["wicket"]["caught"]*wickets["caught"]
        return fielding_points


    def calculate_points(df):
        df["fantasy_points"] = 0
        df["batting_points"] = 0
        df["bowling_points"] = 0
        df["fielding_points"] = 0

        # Convert 'wickettypes' column from string to dictionary
        df['wickettypes'] = df['wickettypes'].apply(ast.literal_eval)

        # Extract wicket types into separate columns
        df['lbw'] = df['wickettypes'].apply(lambda x: x.get('lbw', 0))
        df['bowled'] = df['wickettypes'].apply(
            lambda x: x.get('bowled', 0) + x.get('caught and bowled', 0))
        df['caught'] = df['wickettypes'].apply(lambda x: x.get('caught', 0))
        df['stumped'] = df['wickettypes'].apply(lambda x: x.get('stumped', 0))
        df['run_out'] = df['wickettypes'].apply(lambda x: x.get('run out', 0))

        # Define a function to calculate all points for a row
        def calculate_rowpoints(row):
            wickets = {
                "lbw": row['lbw'],
                "bowled": row['bowled'],
                "caught": row['caught'],
                "stumped": row['stumped'],
                "run out": row['run_out'],
                "catches_taken": row['catches_taken']
            }

            row['fantasy_points'] = calculate_fantasy_points(
                row["match_type"], "batsmen", "playing11", row["runs_scored"], row["fours"], row["sixes"],
                wickets, "player", row["overs_bowled"], row["balls_played"], row["runs_given"], row["maiden_overs"]
            )

            row['batting_points'] = calculate_batting_points(
                row["match_type"], "batsmen", row["runs_scored"], row["fours"], row["sixes"], row["balls_played"]
            )

            row['bowling_points'] = calculate_bowling_points(
                row["match_type"], "batsmen", "playing11", row["runs_scored"], row["fours"], row["sixes"],
                wickets, "player", row["overs_bowled"], row["balls_played"], row["runs_given"], row["maiden_overs"]
            )

            row['fielding_points'] = calculate_fielding_points(
                row["match_type"], "batsmen", "playing11", row["runs_scored"], row["fours"], row["sixes"],
                wickets, "player", row["overs_bowled"], row["balls_played"], row["runs_given"], row["maiden_overs"]
            )

            return row

        # Apply the function to calculate points for each row
        df = df.apply(calculate_rowpoints, axis=1)
        return df


    def process_all_files():
        input_folder = interim_dir
        output_folder = processed_dir

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in os.listdir(input_folder):
            if file.endswith(".csv"):
                file_path = os.path.join(input_folder, file)
                df = pd.read_csv(file_path)
                df_with_points = calculate_points(df)
                output_file_name = os.path.splitext(file)[0] + "_points.csv"
                output_file_path = os.path.join(output_folder, output_file_name)
                df_with_points.to_csv(output_file_path, index=False)


    process_all_files()


if __name__ == "__main__":
    urls = [
        "https://cricsheet.org/downloads/blz_json.zip"
    ]
    raw_dir = "C:/InterIIT/Github Files/Zip Files"
    interim_dir = r"C:\InterIIT\Github Files\Interim Excel Files"
    processed_dir = r"C:\InterIIT\Github Files\Processed Points Excels"

    download_zip_files(urls, raw_dir)
    extract_and_convert_to_csv(raw_dir, interim_dir)
    process_final_csv(interim_dir, processed_dir)