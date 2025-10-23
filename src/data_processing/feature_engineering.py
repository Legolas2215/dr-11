import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
import numpy as np

def calculate_career_stats(df):
    def classify_player(row):
        if row["avg_wkt_taken"] < 0.5 or row["bowling_avg"] < 1:
            return 0  # Batsman
        elif ((row["avg_wkt_taken"] > 0.5) and( row['batting_avg']>10)) and (row["bowling_avg"] > 1):
            return 1  # All
        else:
            return 2  # Bowl

    df['player_id'].nunique()

    matches_played_df = df["player_id"].value_counts().reset_index()
    matches_played_df.columns = ["player_id", "matches_played"]

    # Step 4: Aggregate player statistics for total runs, total balls played, total balls thrown, and total runs given
    career_stats_df = df.groupby("player_id").agg({
        "runs_scored": "sum",
        "balls_played": "sum",
        "balls_thrown": "sum",
        "runs_given": "sum",
        "wickets_taken":"sum"
    }).reset_index()

    # Step 5: Merge the aggregated stats with the number of matches played
    career_stats_df = pd.merge(matches_played_df, career_stats_df, on="player_id")

    # Step 6: Rename columns to reflect career statistics
    career_stats_df.rename(columns={
        "runs_scored": "total_runs",
        "balls_played": "total_balls_played",
        "balls_thrown": "total_balls_thrown",
        "runs_given": "total_runs_given",
        "wickets_taken":"total_wickets_taken"
    }, inplace=True)

    career_stats_df["batting_avg"] = career_stats_df["total_runs"] / career_stats_df["matches_played"]
    career_stats_df["avg_wkt_taken"] = career_stats_df["total_wickets_taken"] / career_stats_df["matches_played"]
    career_stats_df["avg_economy"] = career_stats_df["total_runs_given"]*6 / (career_stats_df["total_balls_thrown"]+1)
    career_stats_df["bowling_avg"] = career_stats_df["total_balls_thrown"] / (6*career_stats_df["matches_played"])


    career_stats_df["player_type"] = career_stats_df.apply(classify_player, axis=1)
    df = df.merge(career_stats_df[['player_id', 'player_type']], on='player_id', how='left')
    df = df.sort_values(by=['player_name', 'date_of_the_match'])
    return df

def convert_test_csv(data):
    match_columns = [
       'date_of_the_match', 'match_type', 'venue',
       'event', 'season', 'toss_winner', 'toss_decision', 'winner',
       'player_name', 'player_id', 'team_name', 'opponent_team','filename','match_number', 'city'
    ]

    stats_to_pivot = [
        'runs_scored','balls_played', 'balls_thrown', 'extras_given', 'fours', 'sixes',
        'catches_taken','player_of_the_match_yes_or_no', 'out_kind', 'runs_given',
        'wickets_taken', 'fours_given', 'sixes_given', 'overs_bowled',
        'dot_balls', 'wickettypes', 'maiden_overs', 'over_faced_first',
        'out_by_bowler', 'out_by_fielder', 'fantasy_points', 'batting_points',
        'bowling_points', 'fielding_points', 'lbw', 'bowled', 'caught',
        'stumped', 'run_out'
    ]
    data['match_player_key'] = data['date_of_the_match'].astype(str) + "_" + data['player_id'].astype(str)
    reshaped_stats = []
    for stat in stats_to_pivot:
        pivot = data.pivot_table(
            index=['match_player_key'], 
            columns='inning_no', 
            values=stat, 
            aggfunc='first'
        )
        pivot.columns = [f"{stat}_in_{inning}" for inning in pivot.columns]
        reshaped_stats.append(pivot)
    stats_data = pd.concat(reshaped_stats, axis=1)
    df = data[match_columns + ['match_player_key']].drop_duplicates().merge(
        stats_data, on='match_player_key', how='left'
    )
    df.drop(columns=['match_player_key'], inplace=True)
    stats = [
        'runs_scored', 'balls_played', 'balls_thrown', 'extras_given', 'fours', 'sixes',
        'catches_taken', 'player_of_the_match_yes_or_no', 'runs_given', 'wickets_taken',
        'fours_given', 'sixes_given', 'overs_bowled', 'dot_balls', 'maiden_overs',
        'over_faced_first',  'fantasy_points', 'batting_points',
        'bowling_points', 'fielding_points', 'lbw', 'bowled', 'caught', 'stumped', 'run_out'
    ]
    for stat in stats:
        df[f'{stat}_in_1'] = df[f'{stat}_in_1'].fillna(0)
        df[f'{stat}_in_2'] = df[f'{stat}_in_2'].fillna(0)

    columns = ['player_name'] + [f'{stat}_in_1' for stat in stats] + [f'{stat}_in_2' for stat in stats]

    for stat in stats:
        df[stat] = df[f'{stat}_in_1'] + df[f'{stat}_in_2']
    cols_to_keep = [
        'date_of_the_match', 'match_type', 'venue', 'event', 'season',
        'toss_winner', 'toss_decision', 'winner', 'player_name', 'player_id',
        'team_name', 'opponent_team', 'filename', 'match_number', 'city','out_kind_in_1','out_kind_in_2',
        'out_by_bowler_in_1', 'out_by_bowler_in_2', 'out_by_fielder_in_1',
        'out_by_fielder_in_2','runs_scored', 'balls_played', 'balls_thrown',
        'extras_given', 'fours', 'sixes', 'catches_taken',
        'player_of_the_match_yes_or_no', 'runs_given', 'wickets_taken',
        'fours_given', 'sixes_given', 'overs_bowled', 'dot_balls',
        'maiden_overs', 'over_faced_first', 'fantasy_points', 'batting_points',
        'bowling_points', 'fielding_points', 'lbw', 'bowled', 'caught',
        'stumped', 'run_out'
    ]

    df = df[cols_to_keep]
    return df

def calculate_career_stats_test(df):
    def classify_player(row):
        if row["avg_wkt_taken"] < 1 or row["bowling_avg"] < 1:
            return 0  # Batsman
        elif ((row["avg_wkt_taken"] > 1.5) and( row['batting_avg']>25)) and (row["bowling_avg"] > 1):
            return 1  # All
        else:
            return 2  # Bowl

    df['player_id'].nunique()

    matches_played_df = df["player_id"].value_counts().reset_index()
    matches_played_df.columns = ["player_id", "matches_played"]

    career_stats_df = df.groupby("player_id").agg({
        "runs_scored": "sum",
        "balls_played": "sum",
        "balls_thrown": "sum",
        "runs_given": "sum",
        "wickets_taken":"sum"
    }).reset_index()

    career_stats_df = pd.merge(matches_played_df, career_stats_df, on="player_id")

    career_stats_df.rename(columns={
        "runs_scored": "total_runs",
        "balls_played": "total_balls_played",
        "balls_thrown": "total_balls_thrown",
        "runs_given": "total_runs_given",
        "wickets_taken":"total_wickets_taken"
    }, inplace=True)

    career_stats_df["batting_avg"] = career_stats_df["total_runs"] / career_stats_df["matches_played"]
    career_stats_df["avg_wkt_taken"] = career_stats_df["total_wickets_taken"] / career_stats_df["matches_played"]
    career_stats_df["avg_economy"] = career_stats_df["total_runs_given"]*6 / (career_stats_df["total_balls_thrown"]+1)
    career_stats_df["bowling_avg"] = career_stats_df["total_balls_thrown"] / (6*career_stats_df["matches_played"])
    career_stats_df["player_type"] = career_stats_df.apply(classify_player, axis=1)

    df = df.merge(career_stats_df[['player_id', 'player_type']], on='player_id', how='left')

    return df
   
def prepare_datasets(df, start_date, end_date):
    df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
    df = df[(df['date_of_the_match'] <= end_date) & (df['date_of_the_match'] >= start_date)]
    return df

def prepare_fantasy_points_shifted(df):
    df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
    df_sorted = df.sort_values(by=['player_id', 'date_of_the_match'], ascending=[True, True]).reset_index(drop=True)
    df_sorted['fantasy_points_shifted'] = df_sorted.groupby('player_id')['fantasy_points'].shift(-1)
    df_sorted['bowling_points_shifted'] = df_sorted.groupby('player_id')['bowling_points'].shift(-1)
    df_sorted['batting_points_shifted'] = df_sorted.groupby('player_id')['batting_points'].shift(-1)
    df_sorted['fielding_points_shifted'] = df_sorted.groupby('player_id')['fielding_points'].shift(-1)
    df_sorted = df_sorted.fillna(method='ffill')
    df_cleaned = df_sorted.fillna(0)
    def remove_first_row(group):
        return group.iloc[:]
    df_final = df_cleaned.groupby('player_id').apply(remove_first_row).reset_index(drop=True)
    return df_final

class TrainFeatures:
    @staticmethod
    def runs_last_n(df, window):
        runs_last_n = (
            df.groupby('player_name')['runs_scored']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['runs_last_n'] = runs_last_n.values
        df['runs_last_n'] = df['runs_last_n'].fillna(0)
        return df

    @staticmethod
    def fours_last_n(df, window):
        fours_last_n = (
            df.groupby('player_name')['fours']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['fours_last_n'] = fours_last_n.values
        df['fours_last_n'] = df['fours_last_n'].fillna(0)
        return df

    @staticmethod
    def fifties_last_n(df, window):
        df['is_50'] = (df['runs_scored'] >= 50).astype(int)
        fifties_last_n = (
            df.groupby('player_name')['is_50']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['fifties_last_n'] = fifties_last_n.values
        df['fifties_last_n'] = df['fifties_last_n'].fillna(0)
        return df

    @staticmethod
    def hundreds_last_n(df, window):
        df['is_100'] = (df['runs_scored'] >= 100).astype(int)
        hundreds_last_n = (
            df.groupby('player_name')['is_100']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['hundreds_last_n'] = hundreds_last_n.values
        df['hundreds_last_n'] = df['hundreds_last_n'].fillna(0)
        return df

    @staticmethod
    def strike_rate_last_n(df, window):
        df['strike_rate'] = (df['runs_scored'] / df['balls_played']) * 100
        df['strike_rate'] = df['strike_rate'].replace([float('inf'), -float('inf')], 0)
        strike_rate_last_n = (
            df.groupby('player_name')['strike_rate']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).mean())
        )
        df['strike_rate_last_n'] = strike_rate_last_n.values
        df['strike_rate_last_n'] = df['strike_rate_last_n'].fillna(0)
        return df

    @staticmethod
    def sixes_last_n(df, window):
        sixes_last_n = (
            df.groupby('player_name')['sixes']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['sixes_last_n'] = sixes_last_n.values
        df['sixes_last_n'] = df['sixes_last_n'].fillna(0)
        return df

    @staticmethod
    def cumulative_average_batting_points(df):
        df['ones_column'] = 1
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['year'] = df['date_of_the_match'].dt.year
        df['cumulative_batting_points'] = df.groupby(['player_id'])['batting_points'].cumsum()
        df['cumulative_matches'] = df.groupby(['player_id'])['ones_column'].cumsum()
        df['cumulative_average_batting'] = df['cumulative_batting_points'] / df['cumulative_matches']
        return df

    @staticmethod
    def balls_thrown_last_n(df, window):
        balls_thrown_last_n = (
            df.groupby('player_name')['balls_thrown']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['balls_thrown_last_n'] = balls_thrown_last_n.values
        df['balls_thrown_last_n'] = df['balls_thrown_last_n'].fillna(0)
        return df

    @staticmethod
    def extras_given_last_n(df, window):
        extras_given_last_n = (
            df.groupby('player_name')['extras_given']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['extras_given_last_n'] = extras_given_last_n.values
        df['extras_given_last_n'] = df['extras_given_last_n'].fillna(0)
        return df

    @staticmethod
    def runs_given_last_n(df, window):
        runs_given_last_n = (
            df.groupby('player_name')['runs_given']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['runs_given_last_n'] = runs_given_last_n.values
        df['runs_given_last_n'] = df['runs_given_last_n'].fillna(0)
        return df

    @staticmethod
    def overs_bowled_last_n(df, window):
        overs_bowled_last_n = (
            df.groupby('player_name')['overs_bowled']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['overs_bowled_last_n'] = overs_bowled_last_n.values
        df['overs_bowled_last_n'] = df['overs_bowled_last_n'].fillna(0)
        return df

    @staticmethod
    def maiden_overs_last_n(df, window):
        maiden_overs_last_n = (
            df.groupby('player_name')['maiden_overs']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['maiden_overs_last_n'] = maiden_overs_last_n.values
        df['maiden_overs_last_n'] = df['maiden_overs_last_n'].fillna(0)
        return df

    @staticmethod
    def economy_last_n(df, window):
        df['economy_rate'] = df['runs_given'] / df['overs_bowled']
        df['economy_rate'] = df['economy_rate'].replace([float('inf'), -float('inf')], 0)
        economy_last_n = (
            df.groupby('player_name')['economy_rate']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).mean())
        )
        df['economy_last_n'] = economy_last_n.values
        df['economy_last_n'] = df['economy_last_n'].fillna(0)
        return df

    @staticmethod
    def wickets_last_n(df, window):
        wickets_last_n = (
            df.groupby('player_name')['wickets_taken']
            .apply(lambda x: x.shift(0).rolling(window=window, min_periods=1).sum())
        )
        df['wickets_last_n'] = wickets_last_n.values
        df['wickets_last_n'] = df['wickets_last_n'].fillna(0)
        return df

    @staticmethod
    def cumulative_average_bowling_points(df):
        df['ones_column'] = 1
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['year'] = df['date_of_the_match'].dt.year
        df['cumulative_fantasy_points'] = df.groupby(['player_id'])['bowling_points'].cumsum()
        df['cumulative_matches'] = df.groupby(['player_id'])['ones_column'].cumsum()
        df['cumulative_average_bowling'] = df['cumulative_fantasy_points'] / df['cumulative_matches']
        return df
    
    @staticmethod
    def days_since_last_match(df):
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['days_since_last_match'] = df.groupby('player_id')['date_of_the_match'].diff().dt.days
        df['days_since_last_match'] = df['days_since_last_match'].fillna(0).astype(int)
        return df

    @staticmethod
    def cumulative_top_11_count(df):
        df['year'] = df['date_of_the_match'].dt.year
        top_11_df = df.groupby('filename').apply(lambda x: x.nlargest(11, 'fantasy_points')).reset_index(drop=True)
        top_11_counts = top_11_df.groupby(['player_id', 'date_of_the_match', 'year', 'filename']).size().reset_index(name='top_11_count')
        df = df.merge(top_11_counts, on=['player_id', 'date_of_the_match', 'year', 'filename'], how='left')
        df['top_11_count'] = df['top_11_count'].fillna(0).astype(int)
        df['cumulative_top_11_count'] = df.groupby(['player_id', 'year'])['top_11_count'].cumsum()
        return df

    @staticmethod
    def process_player_opponent_stats_batsman(df):
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_name', 'opponent_team', 'date_of_the_match'])
        df['runs_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['runs_scored'].cumsum()
        df['fours_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['fours'].cumsum()
        df['sixes_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['sixes'].cumsum()
        return df

    @staticmethod
    def process_player_opponent_stats_bowler(df):
        df['wickets_taken_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['wickets_taken'].cumsum()
        df['cumulative_matches_against_opponent'] = df.groupby(['player_id', 'opponent_team']).cumcount() + 1
        df['cumulative_runs_given'] = df.groupby(['player_id', 'opponent_team'])['runs_given'].cumsum()
        df['cumulative_avg_runs_given'] = df['cumulative_runs_given'] / df['cumulative_matches_against_opponent']
        return df
    
    @staticmethod
    def process_player_venue_stats_batman(df):
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_name', 'venue', 'date_of_the_match'])
        df['runs_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['runs_scored'].cumsum()
        df['fours_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['fours'].cumsum()
        df['sixes_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['sixes'].cumsum()
        return df
    
    @staticmethod
    def process_player_venue_stats_bowler(df):
        df['wickets_taken_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['wickets_taken'].cumsum()
        df['cumulative_matches_venue'] = df.groupby(['player_id', 'venue']).cumcount() + 1
        df['cumulative_runs_given_venue'] = df.groupby(['player_id', 'venue'])['runs_given'].cumsum()
        df['cumulative_avg_runs_given_venue'] = df['cumulative_runs_given_venue'] / df['cumulative_matches_venue']
        return df

    @staticmethod
    def calculate_weighted_scores(df):
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df['weighted_score_last_2_months'] = 0.0
        for player_id, player_df in df.groupby('player_id'):
            player_df = player_df.reset_index(drop=True)
            weighted_scores = []
            for i, row in player_df.iterrows():
                match_date = row['date_of_the_match']
                start_date = match_date - timedelta(days=365)
                last_2_months = player_df[(player_df['date_of_the_match'] >= start_date) & 
                                        (player_df['date_of_the_match'] <= match_date)]
                days_diff = (match_date - last_2_months['date_of_the_match']).dt.days
                weighted_score = last_2_months['runs_scored'] * np.exp(-days_diff / 200)
                weighted_scores.append(weighted_score.sum())
            df.loc[df['player_id'] == player_id, 'weighted_score_last_2_months'] = weighted_scores
        return df
    
    @staticmethod
    def optimized_exponential_decay(df, columns = ['batting_points', 'bowling_points'], decay_rate=0.1):
        df = df.sort_values(by=['player_id', 'date_of_the_match']).reset_index(drop=True)
        results = {col: np.zeros(len(df)) for col in columns}
        for player_id, group in df.groupby('player_id'):
            for col in columns:
                points = group[col].values
                n = len(points)
                cumulative = np.zeros(n)
                for i in range(1, n):
                    weights = np.exp(-decay_rate * np.arange(i))
                    cumulative[i] = np.sum(points[:i] * weights)
                results[col][group.index] = cumulative
        decayed_df = pd.DataFrame(results, index=df.index)
        df[['batting_points_decayed', 'bowling_points_decayed']]  = decayed_df
        return df
    
    @staticmethod
    def compute_exponential_average(df, columns = ['batting_points', 'bowling_points'], span=3):
        required_columns = ['player_id', 'opponent_team', 'date_of_the_match'] + columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The DataFrame is missing the following required columns: {missing_columns}")
        df = df.sort_values(by=['player_id', 'opponent_team', 'date_of_the_match'])
        grouped = df.groupby(['player_id', 'opponent_team'], group_keys=False)
        def compute_ewma(group):
            group = group.copy()

            for col in columns:
                group[f'{col}_lag'] = group[col].shift(0)
                group[f'{col}_exp_avg'] = group[f'{col}_lag'].ewm(span=span, adjust=False).mean()
                if len(group) < 2:
                    group[f'{col}_exp_avg'] = group[f'{col}_exp_avg'].fillna(0)
                else:
                    group[f'{col}_exp_avg'] = group[f'{col}_exp_avg'].fillna(0)
                group = group.drop(columns=[f'{col}_lag'])
            return group
        df = grouped.apply(compute_ewma).reset_index(drop=True)
        return df

    @staticmethod
    def exponential_decay_winning_average(df, decay_rate=0.1):
        df = df.sort_values(by=['player_id', 'date_of_the_match']).reset_index(drop=True)
        result_batting = np.zeros(len(df))
        result_bowling = np.zeros(len(df))
        for player_id, group in df.groupby('player_id'):
            batting_points = group['batting_points'].values  
            bowling_points = group['bowling_points'].values
            player_team = group['team_name'].values
            winner_team = group['winner'].values
            n = len(batting_points)
            cumulative_batting = np.zeros(n)
            cumulative_bowling = np.zeros(n)
            for i in range(1, n):
                past_winning_indices = np.where(winner_team[:i] == player_team[:i])[0]
                if len(past_winning_indices) > 0:
                    weights = np.exp(-decay_rate * (i - past_winning_indices))
                    cumulative_batting[i] = np.sum(batting_points[past_winning_indices] * weights) / np.sum(weights)
                    cumulative_bowling[i] = np.sum(bowling_points[past_winning_indices] * weights) / np.sum(weights)
                else:
                    cumulative_batting[i] = 0
                    cumulative_bowling[i] = 0
            result_batting[group.index] = cumulative_batting
            result_bowling[group.index] = cumulative_bowling
        df['winning_exponentially_decayed_batting_points'] = result_batting
        df['winning_exponentially_decayed_bowling_points'] = result_bowling
        return df
    
    @staticmethod
    def calculate_bps(df):
        df = df.copy()
        aggregate_columns = {
            'wickets_taken': ['wickets_taken', 'wickets_taken'],
            'maiden_overs': ['maiden_overs', 'maiden_overs'],
            'dot_balls': ['dot_balls', 'dot_balls'],
            'runs_given': ['runs_given', 'runs_given']
        }
        for new_col, cols in aggregate_columns.items():
            df[new_col] = df[cols].fillna(0).sum(axis=1)
        essential_columns = ['player_id', 'date_of_the_match', 'wickets_taken', 
                            'maiden_overs', 'dot_balls', 'runs_given']
        missing_cols = [col for col in essential_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following essential columns are missing from the DataFrame: {missing_cols}")
        else:
            print("All essential columns are present.")
        df['wickets_taken'] = df['wickets_taken'].fillna(0)
        df['maiden_overs'] = df['maiden_overs'].fillna(0)
        df['dot_balls'] = df['dot_balls'].fillna(0)
        df['runs_given'] = df['runs_given'].fillna(0)
        epsilon = 1e-6
        df = df.sort_values(by=['player_id', 'date_of_the_match'], ascending=[True, True]).reset_index(drop=True)
        df['BPS'] = (df['wickets_taken'] * 4 + df['maiden_overs'] * 3 + df['dot_balls']) / (df['runs_given'] + epsilon)
        return df

    @staticmethod
    def batting_fatigue_score(df):
        weights = [0.05877676, 0.04043447, -0.01322782, 0.62301173]
        df['batting_workload_avg'] = df.groupby('player_name')['batting_points'].expanding().mean().reset_index(level=0, drop=True)
        df['non_boundary_runs'] = df['runs_scored'] - (df['fours'] * 4 + df['sixes'] * 6)
        df['running_between_wickets_avg'] = df.groupby('player_name')['non_boundary_runs'].expanding().mean().reset_index(level=0, drop=True)
        df['days_since_last_match'] = df.groupby('player_name')['date_of_the_match'].diff().dt.days.fillna(0)
        df['match_frequency_avg'] = df.groupby('player_name')['days_since_last_match'].expanding().mean().reset_index(level=0, drop=True)
        df['dot_balls_avg'] = df.groupby('player_name')['dot_balls'].expanding().mean().reset_index(level=0, drop=True)
        df['fatigue_score'] = (
            weights[0] * df['batting_workload_avg'] +           
            weights[1] * df['running_between_wickets_avg'] +    
            weights[2] * df['match_frequency_avg'] +
            weights[3] * df['dot_balls_avg']                    
        )
        return df
 
class TestFeatures:
    @staticmethod
    def balls_thrown_last_n(df, window):
        balls_thrown_last_n = (
            df.groupby('player_name')['balls_thrown']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['balls_thrown_last_n'] = balls_thrown_last_n.values
        df['balls_thrown_last_n'] = df['balls_thrown_last_n'].fillna(0)
        return df

    @staticmethod
    def extras_given_last_n(df, window):
        extras_given_last_n = (
            df.groupby('player_name')['extras_given']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['extras_given_last_n'] = extras_given_last_n.values
        df['extras_given_last_n'] = df['extras_given_last_n'].fillna(0)
        return df

    @staticmethod
    def runs_given_last_n(df, window):
        runs_given_last_n = (
            df.groupby('player_name')['runs_given']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['runs_given_last_n'] = runs_given_last_n.values
        df['runs_given_last_n'] = df['runs_given_last_n'].fillna(0)
        return df

    @staticmethod
    def overs_bowled_last_n(df, window):
        overs_bowled_last_n = (
            df.groupby('player_name')['overs_bowled']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['overs_bowled_last_n'] = overs_bowled_last_n.values
        df['overs_bowled_last_n'] = df['overs_bowled_last_n'].fillna(0)
        return df

    @staticmethod
    def maiden_overs_last_n(df, window):
        maiden_overs_last_n = (
            df.groupby('player_name')['maiden_overs']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['maiden_overs_last_n'] = maiden_overs_last_n.values
        df['maiden_overs_last_n'] = df['maiden_overs_last_n'].fillna(0)
        return df

    @staticmethod
    def economy_last_n(df, window):
        df['economy_rate'] = df['runs_given'] / df['overs_bowled']
        df['economy_rate'] = df['economy_rate'].replace([float('inf'), -float('inf')], 0)
        economy_last_n = (
            df.groupby('player_name')['economy_rate']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
        df['economy_last_n'] = economy_last_n.values
        df['economy_last_n'] = df['economy_last_n'].fillna(0)
        return df

    @staticmethod
    def wickets_last_n(df, window):
        wickets_last_n = (
            df.groupby('player_name')['wickets_taken']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['wickets_last_n'] = wickets_last_n.values
        df['wickets_last_n'] = df['wickets_last_n'].fillna(0)
        return df

    @staticmethod
    def runs_last_n(df, window):
        runs_last_n = (
            df.groupby('player_name')['runs_scored']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['runs_last_n'] = runs_last_n.values
        df['runs_last_n'] = df['runs_last_n'].fillna(0)
        return df

    @staticmethod
    def fours_last_n(df, window):
        fours_last_n = (
            df.groupby('player_name')['fours']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['fours_last_n'] = fours_last_n.values
        df['fours_last_n'] = df['fours_last_n'].fillna(0)
        return df

    @staticmethod
    def fifties_last_n(df, window):
        df['is_50'] = (df['runs_scored'] >= 50).astype(int)
        fifties_last_n = (
            df.groupby('player_name')['is_50']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['fifties_last_n'] = fifties_last_n.values
        df['fifties_last_n'] = df['fifties_last_n'].fillna(0)
        return df

    @staticmethod
    def hundreds_last_n(df, window):
        df['is_100'] = (df['runs_scored'] >= 100).astype(int)
        hundreds_last_n = (
            df.groupby('player_name')['is_100']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['hundreds_last_n'] = hundreds_last_n.values
        df['hundreds_last_n'] = df['hundreds_last_n'].fillna(0)
        return df

    @staticmethod
    def strike_rate_last_n(df, window):
        df['strike_rate'] = (df['runs_scored'] / df['balls_played']) * 100
        df['strike_rate'] = df['strike_rate'].replace([float('inf'), -float('inf')], 0)
        strike_rate_last_n = (
            df.groupby('player_name')['strike_rate']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
        df['strike_rate_last_n'] = strike_rate_last_n.values
        df['strike_rate_last_n'] = df['strike_rate_last_n'].fillna(0)
        return df

    @staticmethod
    def sixes_last_n(df, window):
        sixes_last_n = (
            df.groupby('player_name')['sixes']
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df['sixes_last_n'] = sixes_last_n.values
        df['sixes_last_n'] = df['sixes_last_n'].fillna(0)
        return df

    @staticmethod
    def cumulative_average_batting_points(df):
        df['ones_column'] = 1
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['cumulative_batting_points'] = df.groupby('player_id')['batting_points'].cumsum()
        df['cumulative_matches'] = df.groupby('player_id')['ones_column'].cumsum()
        df['cumulative_batting_points'] = df.groupby('player_id')['cumulative_batting_points'].shift(1)
        df['cumulative_matches'] = df.groupby('player_id')['cumulative_matches'].shift(1)
        df['cumulative_batting_points'].fillna(0, inplace=True)
        df['cumulative_matches'].fillna(1, inplace=True)
        df['cumulative_average_batting'] = df['cumulative_batting_points'] / df['cumulative_matches']
        return df

    @staticmethod
    def cumulative_average_bowling_points(df):
        df['ones_column'] = 1
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['year'] = df['date_of_the_match'].dt.year
        df['cumulative_bowling_points'] = df.groupby(['player_id'])['bowling_points'].cumsum()
        df['cumulative_matches'] = df.groupby(['player_id'])['ones_column'].cumsum()
        df['cumulative_bowling_points'] = df.groupby(['player_id'])['cumulative_bowling_points'].shift(1)
        df['cumulative_matches'] = df.groupby(['player_id'])['cumulative_matches'].shift(1)
        df['cumulative_bowling_points'].fillna(0, inplace=True)
        df['cumulative_matches'].fillna(1, inplace=True)
        df['cumulative_average_bowling'] = df['cumulative_bowling_points'] / df['cumulative_matches']
        return df
    
    @staticmethod
    def days_since_last_match(df):
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['days_since_last_match'] = df.groupby('player_id')['date_of_the_match'].diff().dt.days
        df['days_since_last_match'] = df['days_since_last_match'].fillna(0).astype(int)
        return df
    
    @staticmethod
    def cumulative_top_11_count(df):
        df['year'] = df['date_of_the_match'].dt.year
        top_11_df = df.groupby('filename').apply(lambda x: x.nlargest(11, 'fantasy_points')).reset_index(drop=True)
        top_11_counts = top_11_df.groupby(['player_id', 'date_of_the_match', 'year', 'filename']).size().reset_index(name='top_11_count')
        df = df.merge(top_11_counts, on=['player_id', 'date_of_the_match', 'year', 'filename'], how='left')
        df['top_11_count'] = df['top_11_count'].fillna(0).astype(int)
        df['cumulative_top_11_count'] = df.groupby(['player_id', 'year'])['top_11_count'].cumsum()
        df['cumulative_top_11_count'] = df.groupby(['player_id', 'year'])['cumulative_top_11_count'].shift(1).fillna(0)
        return df
    
    @staticmethod
    def process_player_opponent_stats_batsman(df):
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_name', 'opponent_team', 'date_of_the_match'])
        df['runs_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['runs_scored'].cumsum()
        df['runs_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['runs_scored_till_date_against_opponent_by_this_player'].shift(1).fillna(0)
        df['fours_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['fours'].cumsum()
        df['fours_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['fours_scored_till_date_against_opponent_by_this_player'].shift(1).fillna(0)
        df['sixes_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['sixes'].cumsum()
        df['sixes_scored_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['sixes_scored_till_date_against_opponent_by_this_player'].shift(1).fillna(0)
        return df
    
    @staticmethod
    def process_player_opponent_stats_bowler(df):
        df['wickets_taken_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['wickets_taken'].cumsum()
        df['wickets_taken_till_date_against_opponent_by_this_player'] = df.groupby(['player_name', 'opponent_team'])['wickets_taken_till_date_against_opponent_by_this_player'].shift(1).fillna(0)
        df['cumulative_matches_against_opponent'] = df.groupby(['player_id', 'opponent_team']).cumcount() + 1
        df['cumulative_matches_against_opponent'] = df.groupby(['player_id', 'opponent_team'])['cumulative_matches_against_opponent'].shift(1).fillna(1)
        df['cumulative_runs_given'] = df.groupby(['player_id', 'opponent_team'])['runs_given'].cumsum()
        df['cumulative_runs_given'] = df.groupby(['player_id', 'opponent_team'])['cumulative_runs_given'].shift(1).fillna(0)
        df['cumulative_avg_runs_given'] = df['cumulative_runs_given'] / df['cumulative_matches_against_opponent']
        return df

    @staticmethod
    def process_player_venue_stats_batsman(df):
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df = df.sort_values(by=['player_name', 'venue', 'date_of_the_match'])
        df['runs_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['runs_scored'].cumsum()
        df['runs_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['runs_scored_till_date_on_this_venue_by_this_player'].shift(1).fillna(0)
        df['fours_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['fours'].cumsum()
        df['fours_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['fours_scored_till_date_on_this_venue_by_this_player'].shift(1).fillna(0)
        df['sixes_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['sixes'].cumsum()
        df['sixes_scored_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['sixes_scored_till_date_on_this_venue_by_this_player'].shift(1).fillna(0)
        return df
    
    @staticmethod
    def process_player_venue_stats_bowler(df):
        df['wickets_taken_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['wickets_taken'].cumsum()
        df['wickets_taken_till_date_on_this_venue_by_this_player'] = df.groupby(['player_name', 'venue'])['wickets_taken_till_date_on_this_venue_by_this_player'].shift(1).fillna(0)
        df['cumulative_matches_venue'] = df.groupby(['player_id', 'venue']).cumcount() + 1
        df['cumulative_matches_venue'] = df.groupby(['player_id', 'venue'])['cumulative_matches_venue'].shift(1).fillna(1)
        df['cumulative_runs_given_venue'] = df.groupby(['player_id', 'venue'])['runs_given'].cumsum()
        df['cumulative_runs_given_venue'] = df.groupby(['player_id', 'venue'])['cumulative_runs_given_venue'].shift(1).fillna(0)
        df['cumulative_avg_runs_given_venue'] = df['cumulative_runs_given_venue'] / df['cumulative_matches_venue']
        return df
    
    @staticmethod
    def calculate_weighted_scores(df):
        df = df.sort_values(by=['player_id', 'date_of_the_match'])
        df['date_of_the_match'] = pd.to_datetime(df['date_of_the_match'])
        df['weighted_score_last_2_months'] = 0.0
        for player_id, player_df in df.groupby('player_id'):
            player_df = player_df.reset_index(drop=True)
            weighted_scores = []
            for i, row in player_df.iterrows():
                match_date = row['date_of_the_match']
                start_date = match_date - timedelta(days=365)
                last_2_months = player_df[(player_df['date_of_the_match'] >= start_date) & 
                                        (player_df['date_of_the_match'] <= match_date)]
                days_diff = (match_date - last_2_months['date_of_the_match']).dt.days
                weighted_score = last_2_months['runs_scored'] * np.exp(-days_diff / 200)
                weighted_scores.append(weighted_score.sum())
            df.loc[df['player_id'] == player_id, 'weighted_score_last_2_months'] = weighted_scores
        df['weighted_score_last_2_months'] = df.groupby(['player_id'])['weighted_score_last_2_months'].shift(1).fillna(0)
        return df
    
    @staticmethod
    def optimized_exponential_decay(df, columns = ['batting_points', 'bowling_points'], decay_rate=0.1):
        df = df.sort_values(by=['player_id', 'date_of_the_match']).reset_index(drop=True)
        results = {col: np.zeros(len(df)) for col in columns}
        for player_id, group in df.groupby('player_id'):
            for col in columns:
                points = group[col].values
                n = len(points)
                cumulative = np.zeros(n)
                for i in range(1, n):
                    weights = np.exp(-decay_rate * np.arange(i))
                    cumulative[i] = np.sum(points[:i] * weights)
                results[col][group.index] = cumulative
        decayed_df = pd.DataFrame(results, index=df.index)
        df[['batting_points_decayed', 'bowling_points_decayed']]  = decayed_df
        df['batting_points_decayed'] = df.groupby('player_id')['batting_points_decayed'].shift(1).fillna(0)
        df['bowling_points_decayed'] = df.groupby('player_id')['bowling_points_decayed'].shift(1).fillna(0)
        return df
    
    @staticmethod
    def compute_exponential_average(df, columns = ['batting_points', 'bowling_points'], span=3):
        required_columns = ['player_id', 'opponent_team', 'date_of_the_match'] + columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The DataFrame is missing the following required columns: {missing_columns}")
        df = df.sort_values(by=['player_id', 'opponent_team', 'date_of_the_match'])
        grouped = df.groupby(['player_id', 'opponent_team'], group_keys=False)
        def compute_ewma(group):
            group = group.copy()

            for col in columns:
                group[f'{col}_lag'] = group[col].shift(0)
                group[f'{col}_exp_avg'] = group[f'{col}_lag'].ewm(span=span, adjust=False).mean()
                if len(group) < 2:
                    group[f'{col}_exp_avg'] = group[f'{col}_exp_avg'].fillna(0)
                else:
                    group[f'{col}_exp_avg'] = group[f'{col}_exp_avg'].fillna(0)
                group = group.drop(columns=[f'{col}_lag'])
            return group
        df = grouped.apply(compute_ewma).reset_index(drop=True)
        df['batting_points_exp_avg'] = df.groupby(['player_id', 'opponent_team'])['batting_points_exp_avg'].shift(1).fillna(0)
        df['bowling_points_exp_avg'] = df.groupby(['player_id', 'opponent_team'])['bowling_points_exp_avg'].shift(1).fillna(0)
        return df
    
    @staticmethod
    def exponential_decay_winning_average(df, decay_rate=0.1):
        df = df.sort_values(by=['player_id', 'date_of_the_match']).reset_index(drop=True)
        result_batting = np.zeros(len(df))
        result_bowling = np.zeros(len(df))
        for player_id, group in df.groupby('player_id'):
            batting_points = group['batting_points'].values  
            bowling_points = group['bowling_points'].values
            player_team = group['team_name'].values
            winner_team = group['winner'].values
            n = len(batting_points)
            cumulative_batting = np.zeros(n)
            cumulative_bowling = np.zeros(n)
            for i in range(1, n):
                past_winning_indices = np.where(winner_team[:i] == player_team[:i])[0]
                if len(past_winning_indices) > 0:
                    weights = np.exp(-decay_rate * (i - past_winning_indices))
                    cumulative_batting[i] = np.sum(batting_points[past_winning_indices] * weights) / np.sum(weights)
                    cumulative_bowling[i] = np.sum(bowling_points[past_winning_indices] * weights) / np.sum(weights)
                else:
                    cumulative_batting[i] = 0
                    cumulative_bowling[i] = 0
            result_batting[group.index] = cumulative_batting
            result_bowling[group.index] = cumulative_bowling
        df['winning_exponentially_decayed_batting_points'] = result_batting
        df['winning_exponentially_decayed_bowling_points'] = result_bowling
        df['winning_exponentially_decayed_batting_points'] = df.groupby('player_id')['winning_exponentially_decayed_batting_points'].shift(1).fillna(0)
        df['winning_exponentially_decayed_bowling_points'] = df.groupby('player_id')['winning_exponentially_decayed_bowling_points'].shift(1).fillna(0)
        return df
    

    @staticmethod
    def calculate_bps(df):
        df = df.copy()
        aggregate_columns = {
            'wickets_taken': ['wickets_taken', 'wickets_taken'],
            'maiden_overs': ['maiden_overs', 'maiden_overs'],
            'dot_balls': ['dot_balls', 'dot_balls'],
            'runs_given': ['runs_given', 'runs_given']
        }
        for new_col, cols in aggregate_columns.items():
            df[new_col] = df[cols].fillna(0).sum(axis=1)
        essential_columns = ['player_id', 'date_of_the_match', 'wickets_taken', 
                            'maiden_overs', 'dot_balls', 'runs_given']
        missing_cols = [col for col in essential_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following essential columns are missing from the DataFrame: {missing_cols}")
        else:
            print("All essential columns are present.")
            df['wickets_taken'] = df['wickets_taken'].fillna(0)
            df['maiden_overs'] = df['maiden_overs'].fillna(0)
            df['dot_balls'] = df['dot_balls'].fillna(0)
            df['runs_given'] = df['runs_given'].fillna(0)
            epsilon = 1e-6
            df = df.sort_values(by=['player_id', 'date_of_the_match'], ascending=[True, True]).reset_index(drop=True)
            df['BPS'] = (df['wickets_taken'] * 4 + df['maiden_overs'] * 3 + df['dot_balls']) / (df['runs_given'] + epsilon)
            df['BPS'] = df.groupby('player_id')['BPS'].shift(1).fillna(0)
        return df
    
    @staticmethod
    def batting_fatigue_score(df):
        weights = [0.05877676, 0.04043447, -0.01322782, 0.62301173]
        df = df.sort_values(by=['player_name', 'date_of_the_match'])
        df['batting_workload_avg'] = df.groupby('player_name')['batting_points'].expanding().mean().reset_index(level=0, drop=True)
        df['non_boundary_runs'] = df['runs_scored'] - (df['fours'] * 4 + df['sixes'] * 6)
        df['running_between_wickets_avg'] = df.groupby('player_name')['non_boundary_runs'].expanding().mean().reset_index(level=0, drop=True)
        df['days_since_last_match'] = df.groupby('player_name')['date_of_the_match'].diff().dt.days.fillna(0)
        df['match_frequency_avg'] = df.groupby('player_name')['days_since_last_match'].expanding().mean().reset_index(level=0, drop=True)
        df['dot_balls_avg'] = df.groupby('player_name')['dot_balls'].expanding().mean().reset_index(level=0, drop=True)
        df['fatigue_score'] = (
            weights[0] * df['batting_workload_avg'] +           
            weights[1] * df['running_between_wickets_avg'] +    
            weights[2] * df['match_frequency_avg'] +
            weights[3] * df['dot_balls_avg']                    
        )
        df['fatigue_score'] = df.groupby('player_id')['fatigue_score'].shift(1).fillna(0)
        return df


class TargetBatting(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=['batting_points_shifted'], errors='ignore')

class TargetBowling(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=['bowling_points_shifted'], errors='ignore')

class TargetFantasy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=['batting_points_shifted', 'bowling_points_shifted'], errors='ignore')
    
class Batting_Single_Day(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TrainFeatures.runs_last_n(df, 10)
        df = TrainFeatures.fours_last_n(df, 10)
        df = TrainFeatures.fifties_last_n(df, 10)
        df = TrainFeatures.hundreds_last_n(df, 10)
        df = TrainFeatures.strike_rate_last_n(df, 10)
        df = TrainFeatures.sixes_last_n(df, 10)
        df = TrainFeatures.cumulative_average_batting_points(df)
        df = TrainFeatures.days_since_last_match(df)
        df = TrainFeatures.cumulative_top_11_count(df)
        df = TrainFeatures.calculate_weighted_scores(df)
        df = TrainFeatures.process_player_venue_stats_batman(df)
        df = TrainFeatures.process_player_opponent_stats_batsman(df)
        return df

class Bowling_Single_Day(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TrainFeatures.balls_thrown_last_n(df, 10)
        df = TrainFeatures.extras_given_last_n(df, 10)
        df = TrainFeatures.runs_given_last_n(df, 10)
        df = TrainFeatures.overs_bowled_last_n(df, 10)
        df = TrainFeatures.maiden_overs_last_n(df, 10)
        df = TrainFeatures.economy_last_n(df, 10)
        df = TrainFeatures.wickets_last_n(df, 10)
        df = TrainFeatures.cumulative_average_bowling_points(df)
        df = TrainFeatures.days_since_last_match(df)
        df = TrainFeatures.cumulative_top_11_count(df)
        df = TrainFeatures.process_player_venue_stats_bowler(df)
        df = TrainFeatures.process_player_opponent_stats_bowler(df)
        return df

class All_Rounder_Single_Day(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TrainFeatures.runs_last_n(df, 10)
        df = TrainFeatures.fours_last_n(df, 10)
        df = TrainFeatures.fifties_last_n(df, 10)
        df = TrainFeatures.hundreds_last_n(df, 10)
        df = TrainFeatures.strike_rate_last_n(df, 10)
        df = TrainFeatures.sixes_last_n(df, 10)
        df = TrainFeatures.cumulative_average_batting_points(df)
        df = TrainFeatures.balls_thrown_last_n(df, 10)
        df = TrainFeatures.extras_given_last_n(df, 10)
        df = TrainFeatures.runs_given_last_n(df, 10)
        df = TrainFeatures.overs_bowled_last_n(df, 10)
        df = TrainFeatures.maiden_overs_last_n(df, 10)
        df = TrainFeatures.economy_last_n(df, 10)
        df = TrainFeatures.wickets_last_n(df, 10)
        df = TrainFeatures.cumulative_average_bowling_points(df)
        df = TrainFeatures.days_since_last_match(df)
        df = TrainFeatures.cumulative_top_11_count(df)
        df = TrainFeatures.calculate_weighted_scores(df)
        df = TrainFeatures.process_player_venue_stats_batman(df)
        df = TrainFeatures.process_player_opponent_stats_batsman(df)
        df = TrainFeatures.process_player_venue_stats_bowler(df)
        df = TrainFeatures.process_player_opponent_stats_bowler(df)
        return df

class MultiDay_Batting_Features(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TrainFeatures.runs_last_n(df, 10)
        df = TrainFeatures.fours_last_n(df, 10)
        df = TrainFeatures.fifties_last_n(df, 10)
        df = TrainFeatures.hundreds_last_n(df, 10)
        df = TrainFeatures.sixes_last_n(df, 10)
        df = TrainFeatures.cumulative_average_batting_points(df)
        df = TrainFeatures.days_since_last_match(df)
        df = TrainFeatures.cumulative_top_11_count(df)
        df = TrainFeatures.calculate_weighted_scores(df)
        df = TrainFeatures.process_player_venue_stats_batman(df)
        df = TrainFeatures.process_player_opponent_stats_batsman(df)
        df = TrainFeatures.optimized_exponential_decay(df)
        df = TrainFeatures.compute_exponential_average(df)
        df = TrainFeatures.exponential_decay_winning_average(df)
        df = TrainFeatures.batting_fatigue_score(df)
        return df
    
class MultiDay_Bowling_Features(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TrainFeatures.balls_thrown_last_n(df, 10)
        df = TrainFeatures.extras_given_last_n(df, 10)
        df = TrainFeatures.runs_given_last_n(df, 10)
        df = TrainFeatures.overs_bowled_last_n(df, 10)
        df = TrainFeatures.maiden_overs_last_n(df, 10)
        df = TrainFeatures.wickets_last_n(df, 10)
        df = TrainFeatures.cumulative_average_bowling_points(df)
        df = TrainFeatures.days_since_last_match(df)
        df = TrainFeatures.cumulative_top_11_count(df)
        df = TrainFeatures.process_player_venue_stats_bowler(df)
        df = TrainFeatures.process_player_opponent_stats_bowler(df)
        df = TrainFeatures.optimized_exponential_decay(df)
        df = TrainFeatures.compute_exponential_average(df)
        df = TrainFeatures.exponential_decay_winning_average(df)
        df = TrainFeatures.calculate_bps(df)
        return df

class MultiDay_AllRounder_Features(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TrainFeatures.runs_last_n(df, 10)
        df = TrainFeatures.fours_last_n(df, 10)
        df = TrainFeatures.fifties_last_n(df, 10)
        df = TrainFeatures.hundreds_last_n(df, 10)
        df = TrainFeatures.sixes_last_n(df, 10)
        df = TrainFeatures.cumulative_average_batting_points(df)
        df = TrainFeatures.balls_thrown_last_n(df, 10)
        df = TrainFeatures.extras_given_last_n(df, 10)
        df = TrainFeatures.runs_given_last_n(df, 10)
        df = TrainFeatures.overs_bowled_last_n(df, 10)
        df = TrainFeatures.maiden_overs_last_n(df, 10)
        df = TrainFeatures.wickets_last_n(df, 10)
        df = TrainFeatures.cumulative_average_bowling_points(df)
        df = TrainFeatures.days_since_last_match(df)
        df = TrainFeatures.cumulative_top_11_count(df)
        df = TrainFeatures.calculate_weighted_scores(df)
        df = TrainFeatures.process_player_venue_stats_batman(df)
        df = TrainFeatures.process_player_opponent_stats_batsman(df)
        df = TrainFeatures.process_player_venue_stats_bowler(df)
        df = TrainFeatures.process_player_opponent_stats_bowler(df)
        df = TrainFeatures.optimized_exponential_decay(df)
        df = TrainFeatures.exponential_decay_winning_average(df)
        df = TrainFeatures.compute_exponential_average(df)
        df = TrainFeatures.calculate_bps(df)
        df = TrainFeatures.batting_fatigue_score(df)
        return df

class MultiDay_Batting_Features_Test(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TestFeatures.runs_last_n(df, 10)
        df = TestFeatures.fours_last_n(df, 10)
        df = TestFeatures.fifties_last_n(df, 10)
        df = TestFeatures.hundreds_last_n(df, 10)
        df = TestFeatures.sixes_last_n(df, 10)
        df = TestFeatures.cumulative_average_batting_points(df)
        df = TestFeatures.days_since_last_match(df)
        df = TestFeatures.cumulative_top_11_count(df)
        df = TestFeatures.calculate_weighted_scores(df)
        df = TestFeatures.process_player_venue_stats_batsman(df)
        df = TestFeatures.process_player_opponent_stats_batsman(df)
        df = TestFeatures.optimized_exponential_decay(df)
        df = TestFeatures.compute_exponential_average(df)
        df = TestFeatures.exponential_decay_winning_average(df)
        df = TestFeatures.batting_fatigue_score(df)
        return df
    
class MultiDay_Bowling_Features_Test(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TestFeatures.balls_thrown_last_n(df, 10)
        df = TestFeatures.extras_given_last_n(df, 10)
        df = TestFeatures.runs_given_last_n(df, 10)
        df = TestFeatures.overs_bowled_last_n(df, 10)
        df = TestFeatures.maiden_overs_last_n(df, 10)
        df = TestFeatures.wickets_last_n(df, 10)
        df = TestFeatures.cumulative_average_bowling_points(df)
        df = TestFeatures.days_since_last_match(df)
        df = TestFeatures.cumulative_top_11_count(df)
        df = TestFeatures.process_player_venue_stats_bowler(df)
        df = TestFeatures.process_player_opponent_stats_bowler(df)
        df = TestFeatures.optimized_exponential_decay(df)
        df = TestFeatures.compute_exponential_average(df)
        df = TestFeatures.exponential_decay_winning_average(df)
        df = TestFeatures.calculate_bps(df)
        return df
    
class MultiDay_AllRounder_Features_Test(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TestFeatures.runs_last_n(df, 10)
        df = TestFeatures.fours_last_n(df, 10)
        df = TestFeatures.fifties_last_n(df, 10)
        df = TestFeatures.hundreds_last_n(df, 10)
        df = TestFeatures.sixes_last_n(df, 10)
        df = TestFeatures.cumulative_average_batting_points(df)
        df = TestFeatures.balls_thrown_last_n(df, 10)
        df = TestFeatures.extras_given_last_n(df, 10)
        df = TestFeatures.runs_given_last_n(df, 10)
        df = TestFeatures.overs_bowled_last_n(df, 10)
        df = TestFeatures.maiden_overs_last_n(df, 10)
        df = TestFeatures.wickets_last_n(df, 10)
        df = TestFeatures.cumulative_average_bowling_points(df)
        df = TestFeatures.days_since_last_match(df)
        df = TestFeatures.cumulative_top_11_count(df)
        df = TestFeatures.calculate_weighted_scores(df)
        df = TestFeatures.process_player_venue_stats_batsman(df)
        df = TestFeatures.process_player_opponent_stats_batsman(df)
        df = TestFeatures.process_player_venue_stats_bowler(df)
        df = TestFeatures.process_player_opponent_stats_bowler(df)
        df = TestFeatures.optimized_exponential_decay(df)
        df = TestFeatures.compute_exponential_average(df)
        df = TestFeatures.exponential_decay_winning_average(df)
        df = TestFeatures.calculate_bps(df)
        df = TestFeatures.batting_fatigue_score(df)
        return df

class Batting_Test_Single(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TestFeatures.runs_last_n(df, 10)
        df = TestFeatures.fours_last_n(df, 10)
        df = TestFeatures.fifties_last_n(df, 10)
        df = TestFeatures.hundreds_last_n(df, 10)
        df = TestFeatures.strike_rate_last_n(df, 10)
        df = TestFeatures.sixes_last_n(df, 10)
        df = TestFeatures.cumulative_average_batting_points(df)
        df = TestFeatures.days_since_last_match(df)
        df = TestFeatures.cumulative_top_11_count(df)
        df = TestFeatures.calculate_weighted_scores(df)
        df = TestFeatures.process_player_venue_stats_batsman(df)
        df = TestFeatures.process_player_opponent_stats_batsman(df)
        return df
    

class Bowling_Test_Single(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TestFeatures.balls_thrown_last_n(df, 10)
        df = TestFeatures.extras_given_last_n(df, 10)
        df = TestFeatures.runs_given_last_n(df, 10)
        df = TestFeatures.overs_bowled_last_n(df, 10)
        df = TestFeatures.maiden_overs_last_n(df, 10)
        df = TestFeatures.economy_last_n(df, 10)
        df = TestFeatures.wickets_last_n(df, 10)
        df = TestFeatures.days_since_last_match(df)
        df = TestFeatures.cumulative_top_11_count(df)
        df = TestFeatures.cumulative_average_bowling_points(df)
        df = TestFeatures.process_player_venue_stats_bowler(df)
        df = TestFeatures.process_player_opponent_stats_bowler(df)
        return df


class All_Rounder_Test_Single(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = TestFeatures.runs_last_n(df, 10)
        df = TestFeatures.fours_last_n(df, 10)
        df = TestFeatures.fifties_last_n(df, 10)
        df = TestFeatures.hundreds_last_n(df, 10)
        df = TestFeatures.strike_rate_last_n(df, 10)
        df = TestFeatures.sixes_last_n(df, 10)
        df = TestFeatures.cumulative_average_batting_points(df)
        df = TestFeatures.balls_thrown_last_n(df, 10)
        df = TestFeatures.extras_given_last_n(df, 10)
        df = TestFeatures.runs_given_last_n(df, 10)
        df = TestFeatures.overs_bowled_last_n(df, 10)
        df = TestFeatures.maiden_overs_last_n(df, 10)
        df = TestFeatures.economy_last_n(df, 10)
        df = TestFeatures.wickets_last_n(df, 10)
        df = TestFeatures.cumulative_average_bowling_points(df)
        df = TestFeatures.days_since_last_match(df)
        df = TestFeatures.cumulative_top_11_count(df)
        df = TestFeatures.calculate_weighted_scores(df)
        df = TestFeatures.process_player_venue_stats_batsman(df)
        df = TestFeatures.process_player_opponent_stats_batsman(df)
        df = TestFeatures.process_player_venue_stats_bowler(df)
        df = TestFeatures.process_player_opponent_stats_bowler(df)
        return df