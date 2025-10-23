from data_processing.feature_engineering import *
import pandas as pd
import numpy as np 
import joblib
import os
import yaml
from api.pipeline import *
from utils.constants import *
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings('ignore')

def get_top11_fantasy_points(group):
    top11 = group.nlargest(11, 'y_pred')['y_pred'].values
    top11 = np.sort(top11)[::-1]
    return pd.Series(top11, index=[f'player_{i+1}_fantasy_points' for i in range(11)])

def get_top11_sums(group):
    top11_fantasy_points = group.nlargest(11, 'fantasy_points_shifted')['fantasy_points_shifted'].sum()
    top11_y_pred = group.nlargest(11, 'y_pred')['y_pred'].sum()
    difference = top11_fantasy_points - top11_y_pred
    return pd.Series({
        'sum_top_11_fantasy_points_shifted': top11_fantasy_points,
        'sum_top_11_y_pred': top11_y_pred,
        'difference': difference
    })

def get_top11_and_rank(group):
    top11 = group.nlargest(11, 'fantasy_points_shifted').copy()
    top11.sort_values('fantasy_points_shifted', ascending=False, inplace=True)
    top11['player_index'] = range(1, len(top11) + 1)
    top11['difference'] = top11['fantasy_points_shifted'].diff(-1).abs()
    top11['difference'] = top11['difference'].fillna(0)
    return top11

def load_hyperparameters(yaml_file, model_name):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    return params[model_name]

class ModelTrainer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.folder_path = os.path.join(base_dir, 'src', 'data', 'processed')
        self.model_artifacts_path = os.path.join(base_dir, 'src', 'model_artifacts')
        self.yaml_file = os.path.join(base_dir, 'src', 'configs','model.yaml')
        self.pipeline_manager = PipelineManager()

    def alpha_train(self, Format, df, type):
        if type == 2:
            alpha_model_path = os.path.join(self.model_artifacts_path, f'{Format}_alpha_product.pkl')
            if os.path.exists(alpha_model_path):
                print(f"Alpha model already exists for {Format}. Skipping training.")
                return
        else:
            alpha_model_path = os.path.join(self.model_artifacts_path, f'{Format}_alpha_model.pkl')
        print(f"Training Alpha for {Format}...")
        X_train = df.groupby('filename').apply(get_top11_fantasy_points).reset_index(drop=True)
        sums_df = df.groupby('filename').apply(get_top11_sums).reset_index(drop=True)
        y_train = sums_df['difference']
        best_params = load_hyperparameters(self.yaml_file, Format + '_alpha_xgb')
        best_model = XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, os.path.join(self.model_artifacts_path, alpha_model_path))

        print(f"Alpha training completed for {Format}.")
        top11_df = df.groupby('filename').apply(get_top11_and_rank).reset_index(drop=True)
        average_difference = top11_df.groupby('player_index')['difference'].mean().reset_index()
        if type == 2:
            average_difference.to_csv(f"{Format}_average_difference_product.csv", index=False)
        else:
            average_difference.to_csv(f"{Format}_average_difference_model.csv", index=False)

    def model_train(self, start_date, end_date):
        print("Training Model UI...")
        for file in os.listdir(self.folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.folder_path, file)
                df = pd.read_csv(file_path)
                Format = file.split('.')[0]
                if Format == 'multi_day':
                    df = convert_test_csv(df)
                    df = calculate_career_stats_test(df)
                else:
                    df = calculate_career_stats(df)
                df = prepare_fantasy_points_shifted(df)
                df = df.sort_values(['date_of_the_match', 'filename'])
                df = prepare_datasets(df, start_date, end_date)

                iso_forest = IsolationForest(contamination=0.4, random_state=42)
                numerical_columns = df.select_dtypes(include=[np.number])
                numerical_columns.drop(['fantasy_points_shifted', 'batting_points_shifted', 'bowling_points_shifted','fielding_points_shifted'], axis=1, inplace=True)
                iso_forest.fit(numerical_columns.fillna(0))
                joblib.dump(iso_forest, os.path.join(self.model_artifacts_path, f'{Format}_isolation_forest_model.pkl'))
                anomaly_scores = iso_forest.decision_function(numerical_columns.fillna(0))
                df['anomaly_score'] = anomaly_scores

                results = pd.DataFrame()
                for player_type, model_name in zip([0, 1, 2], ['batsman', 'allrounder', 'bowler']):
                    player_df = df[df['player_type'] == player_type]
                    if not player_df.empty:
                        pipeline = self.pipeline_manager.get_pipeline(Format)[player_type]
                        pipeline.fit(player_df, player_df[target[player_type]] - (2 - player_type)*(player_type)*player_df['fielding_points_shifted'])
                        model = pipeline.named_steps['regressor']
                        joblib.dump(model, os.path.join(self.model_artifacts_path, f'{model_name}_{Format}.pkl'))
                        transformation = pipeline.named_steps['add_target_feature']
                        player_df = transformation.fit_transform(player_df)
                        X = player_df[features[Format][player_type]]
                        player_df['y_pred'] = model.predict(X)
                        player_df = player_df[['filename', 'player_name', 'y_pred', 'fantasy_points_shifted']]
                        results = pd.concat([results, player_df])
                print(f"Model UI training completed for {Format}.")
                self.alpha_train(Format, results, 1)
        print("Model UI training completed.")


    def product_train(self, Format):
        print("Training Product UI...")
        file_path = os.path.join(self.folder_path, Format + '.csv')
        df = pd.read_csv(file_path)

        if Format == 'multi_day':
            df = convert_test_csv(df)
            df = calculate_career_stats_test(df)
        else:
            df = calculate_career_stats(df)
        df = prepare_fantasy_points_shifted(df)
        df = df.sort_values(['date_of_the_match', 'filename'])

        iso_forest = IsolationForest(contamination=0.4, random_state=42)
        numerical_columns = df.select_dtypes(include=[np.number])
        numerical_columns.drop(['fantasy_points_shifted', 'batting_points_shifted', 'bowling_points_shifted', 'fielding_points_shifted'], axis=1, inplace=True)
        iso_forest.fit(numerical_columns.fillna(0))
        joblib.dump(iso_forest, os.path.join(self.model_artifacts_path, f'{Format}_isolation_forest_product.pkl'))
        anomaly_scores = iso_forest.decision_function(numerical_columns.fillna(0))
        df['anomaly_score'] = anomaly_scores
        print("Done with anomaly detection.")
        results = pd.DataFrame()
        for player_type, model_name in zip([0, 1, 2], ['batsman', 'allrounder', 'bowler']):
            print(f"Training for {model_name} in {Format}")
            player_df = df[df['player_type'] == player_type]
            if not player_df.empty:
                pipeline = self.pipeline_manager.get_pipeline(Format)[player_type]
                pipeline.fit(player_df, player_df[target[player_type]] - (2 - player_type)*(player_type)*player_df['fielding_points_shifted'])
                model = pipeline.named_steps['regressor']
                joblib.dump(model, os.path.join(self.model_artifacts_path, f'{Format}_{model_name}.pkl'))
                transformation = pipeline.named_steps['add_target_feature']
                player_df = transformation.fit_transform(player_df)
                X = player_df[features[Format][player_type]]
                player_df['y_pred'] = model.predict(X)
                player_df = player_df[['filename', 'player_name', 'y_pred', 'fantasy_points_shifted']]
                results = pd.concat([results, player_df])
                print(f"DONE: {model_name}")
        self.alpha_train(Format, results, 2)
        print("Product UI training completed.")