from data_processing.feature_engineering import *
import pandas as pd
import numpy as np
import joblib
import os
from utils.constants import *
from api.pipeline import PipelineManager

import warnings
warnings.filterwarnings('ignore')

def get_top11_fantasy_points(group):
    top11 = group.nlargest(11, 'y_pred')['y_pred'].values

    top11 = np.sort(top11)[::-1]
    return pd.Series(top11, index=[f'player_{i+1}_fantasy_points' for i in range(11)])

def get_pred_players(df, filename):
    top_players = df.nlargest(12, 'y_pred')[['player_name', 'y_pred', 'team_name']]
    top_players.sort_values(by='y_pred', ascending=False, inplace=True)
    
    row = df['date_of_the_match'].unique().tolist()
    teams = df['team_name'].unique()
    row.append(teams[0])
    row.append(teams[1])
    sum = 0

    if len(top_players['team_name'][:11].unique()) == 1:
        top_players.iloc[10] = top_players.iloc[11]

    for _, player in top_players[:11].iterrows():
        row.append(player['player_name'])
        row.append(player['y_pred'])
        sum += player['y_pred']
    
    row.append(sum)
    return row

def get_top_players(df, filename):
    top_players = df.nlargest(12, 'fantasy_points')[['player_name', 'fantasy_points', 'team_name']]
    top_players.sort_values(by='fantasy_points', ascending=False, inplace=True)
    
    row = []
    sum = 0

    if len(top_players['team_name'][:11].unique()) == 1:
        top_players.iloc[10] = top_players.iloc[11]

    for _, player in top_players[:11].iterrows():
        row.append(player['player_name'])
        row.append(player['fantasy_points'])
        sum += player['fantasy_points']
    
    row.append(sum)
    return row



class ModelPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.folder_path = os.path.join(base_dir, 'src', 'data', 'processed')
        self.model_artifacts_path = os.path.join(base_dir, 'src', 'model_artifacts')
        self.pipeline_manager = PipelineManager()

    def alpha_predict(self, Format, df, type):
        print(f"Predicting Alpha for {Format}...")

        X_test = pd.DataFrame()
        filenames = df['filename'].unique()
        for filename in filenames:
            df_ = df[df['filename'] == filename]
            X = df_.groupby('filename').apply(get_top11_fantasy_points).reset_index()
            X_test = pd.concat([X_test, X], axis=0)

        if type == 2:
            model_path = os.path.join(self.model_artifacts_path, f'{Format}_alpha_product.pkl')
        else:
            model_path = os.path.join(self.model_artifacts_path, f'{Format}_alpha_model.pkl')
        model = joblib.load(model_path)
        y_pred = model.predict(X_test.drop(['filename'], axis=1))

        df_pred = pd.DataFrame({'filename': X_test['filename'], 'y_pred': y_pred})
        average_difference = pd.read_csv(f"{Format}_average_difference.csv")
        total = average_difference['difference'].sum()
        print(f"Distributing the values for {Format}...")
        if type == 1:
            os.remove(model_path)
        increments_dict = {}
        for filename in df['filename'].unique():
            df_ = df[df['filename'] == filename].copy()
            df_ = df_.sort_values('y_pred', ascending=False).reset_index(drop=True)
            alpha = df_pred[df_pred['filename'] == filename]['y_pred'].values[0]
            increments = [(average_difference.loc[i, 'difference'] / total) * alpha for i in range(11)]
            increments_dict[filename] = increments

        def apply_increments(group):
            filename = group['filename'].iloc[0]
            increments = increments_dict[filename]
            group = group.sort_values('y_pred', ascending=False).reset_index(drop=True)
            for i in range(min(11, len(group))):
                group.at[i, 'y_pred'] += increments[i]
            return group
        df = df.groupby('filename').apply(apply_increments).reset_index(drop=True)
        if type == 1:
            os.remove(f"{Format}_average_difference_model.csv")
        return df
    
    def predict_model_ui(self, start_date, end_date):
        columns = ['Match Date', 'Team 1', 'Team 2']
        for i in range(1, 12):
            columns.append(f'Predicted Player {i}')
            columns.append(f'Predicted Player {i} Points')
        columns.append('Total Points Predicted')
        for i in range(1, 12):
            columns.append(f'Dream Team Player {i}')
            columns.append(f'Dream Team Player {i} Points')
        columns.append('Total Dream Team Points')
        columns.append('Total Points MAE')

        print("Predicting Model UI...")

        all_results = pd.DataFrame(columns=columns)
        for file in os.listdir(self.folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.folder_path, file)
                df = pd.read_csv(file_path)
                Format = file.split('.')[0]
                Format = 'test_' + Format
                if Format == 'test_multi_day':
                    df = convert_test_csv(df)
                    df = calculate_career_stats_test(df)
                else:
                    df = calculate_career_stats(df)
                df = df.sort_values(['date_of_the_match', 'filename'])
                df = prepare_datasets(df, start_date, end_date)
                file_df = pd.DataFrame()

                model_path = os.path.join(self.model_artifacts_path, f'{Format[5:]}_isolation_forest_model.pkl')
                iso_forest = joblib.load(model_path)
                numerical_columns = df.select_dtypes(include=[np.number])
                anomaly_scores = iso_forest.decision_function(numerical_columns.fillna(0))
                df['anomaly_score'] = anomaly_scores
                os.remove(model_path)

                for player_type, model_name in zip([0, 1, 2], ['batsman', 'allrounder', 'bowler']):
                    print(f"Predicting for {model_name} in {file}")
                    player_df = df[df['player_type'] == player_type]
                    if not player_df.empty:
                        pipeline = self.pipeline_manager.get_pipeline(Format)[player_type]
                        player_df = pipeline.fit_transform(player_df)
                        model_path = os.path.join(self.model_artifacts_path, f'{model_name}_{Format[5:]}.pkl')
                        model = joblib.load(model_path)
                        X = player_df[features[Format[5:]][player_type]]
                        os.remove(model_path)
                        predictions = model.predict(X)
                        player_df['y_pred'] = predictions
                        player_df = player_df[['filename', 'player_name', 'y_pred', 'team_name', 'fantasy_points', 'date_of_the_match']]
                        file_df = pd.concat([file_df, player_df], ignore_index=True)
                    print(f"Done Predicting for {model_name} in {file}")
                
                file_df = self.alpha_predict(Format[5:], file_df, 1)
                results = pd.DataFrame(columns=columns)
                filenames = df['filename'].unique()
                for filename in filenames:
                    df_filtered = file_df[file_df['filename'] == filename]
                    pred_players_row = get_pred_players(df_filtered, filename)
                    top_players_row = get_top_players(df_filtered, filename)
                    combined_row = pred_players_row + top_players_row
                    combined_row.append(top_players_row[-1] - pred_players_row[-1])
                    results.loc[len(results)] = combined_row

                all_results = pd.concat([all_results, results], ignore_index=True)
                print("Done with predictions for", file)
        return all_results
    

    def predict_product_ui(self, Format, date, team1, team2, player_list1, player_list2):
        print("Predicting Product UI...")
        all_results = pd.DataFrame()
        file_path = os.path.join(self.folder_path, Format + '.csv')
        df = pd.read_csv(file_path)

        Format = "test_" + Format
        if Format == 'test_multi_day':
            df = convert_test_csv(df)
            df = calculate_career_stats_test(df)
        else:
            df = calculate_career_stats(df)
        df = df[df['player_id'].isin(player_list1 + player_list2)]
        df = df[df['date_of_the_match'] <= date]
        results = pd.DataFrame()

        iso_forest = joblib.load(os.path.join(self.model_artifacts_path, f'{Format[5:]}_isolation_forest_product.pkl'))
        numerical_columns = df.select_dtypes(include=[np.number])
        anomaly_scores = iso_forest.decision_function(numerical_columns.fillna(0))
        df['anomaly_score'] = anomaly_scores

        for player_type, model_name in zip([0, 1, 2], ['batsman', 'allrounder', 'bowler']):
            player_df = df[df['player_type'] == player_type]
            if not player_df.empty:
                pipeline = self.pipeline_manager.get_pipeline(Format)[player_type]
                player_df = pipeline.fit_transform(player_df)
                player_df = player_df.sort_values(by=['player_id', 'date_of_the_match']).groupby('player_id').tail(1)
                model_path = os.path.join(self.model_artifacts_path, f'{Format[5:]}_{model_name}.pkl')
                model = joblib.load(model_path)
                X = player_df[features[Format[5:]][player_type]]
                predictions = model.predict(X)
                player_df['y_pred'] = predictions
                player_df = player_df[['filename', 'player_name', 'player_id', 'y_pred', 'fantasy_points']]
                results = pd.concat([results, player_df], ignore_index=True)
                print(f"Done Predicting for {model_name} in {Format[5:]}")

        all_results = results[['filename', 'player_name', 'player_id', 'y_pred', 'fantasy_points']]
        all_results['filename'] = "0000000.json"
        all_results = self.alpha_predict(Format[5:], all_results, 2)
        print(f"Done with predictions for the match between, {team1} and {team2}")
        return all_results


