# TODO: sklearn pipeline to preprocess data apply transformations and fit ml model train/predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from data_processing.feature_engineering import *
import lightgbm as lgb
import os
from utils.constants import *
import xgboost as xgb
from sklearn.impute import SimpleImputer

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_artifacts_path = os.path.join(base_dir, 'src', 'model_artifacts')

class PipelineManager:
    def __init__(self):
        self.pipelines = {
            "T20": self.t20_pipeline(),
            "ODI": self.odi_pipeline(),
            "multi_day": self.multi_day_pipeline(),
            "test_T20": self.t20_test_pipeline(),
            "test_ODI": self.odi_test_pipeline(),
            "test_multi_day": self.multi_day_test_pipeline(),
        }
    def t20_pipeline(self):
        return {
                0 : Pipeline([
                    ('add_target_feature', Batting_Single_Day()),
                    ('drop_target_feature', TargetBatting()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['T20'][0])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ]),
                1 : Pipeline([
                    ('add_target_feature', All_Rounder_Single_Day()),
                    ('drop_target_feature', TargetFantasy()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['T20'][1])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ]),
                2 : Pipeline([
                    ('add_target_feature', Bowling_Single_Day()),
                    ('drop_target_feature', TargetBowling()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['T20'][2])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ])
        }

    def t20_test_pipeline(self):
        return {
                0 : Pipeline([
                    ('add_target_feature', Batting_Test_Single())
                ]),
                1 : Pipeline([
                    ('add_target_feature', All_Rounder_Test_Single())
                ]),
                2 : Pipeline([
                    ('add_target_feature', Bowling_Test_Single())
                ])
        }
    
    def multi_day_pipeline(self):
        return {
                0 : Pipeline([
                    ('add_target_feature', MultiDay_Batting_Features()),
                    ('drop_target_feature', TargetBatting()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['multi_day'][0])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ]),
                1 : Pipeline([
                    ('add_target_feature', MultiDay_AllRounder_Features()),
                    ('drop_target_feature', TargetFantasy()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['multi_day'][1])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ]),
                2 : Pipeline([
                    ('add_target_feature', MultiDay_Bowling_Features()),
                    ('drop_target_feature', TargetBowling()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['multi_day'][2])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ])
        }

    def multi_day_test_pipeline(self):
        return {
                0 : Pipeline([
                    ('add_target_feature', MultiDay_Batting_Features_Test())
                ]),
                1 : Pipeline([
                    ('add_target_feature', MultiDay_AllRounder_Features_Test())
                ]),
                2 : Pipeline([
                    ('add_target_feature', MultiDay_Bowling_Features_Test())
                ])
        }
    
    def odi_pipeline(self):
        return {
                0 : Pipeline([
                    ('add_target_feature', Batting_Single_Day()),
                    ('drop_target_feature', TargetBatting()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['ODI'][0])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ]),
                1 : Pipeline([
                    ('add_target_feature', All_Rounder_Single_Day()),
                    ('drop_target_feature', TargetFantasy()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['ODI'][1])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ]),
                2 : Pipeline([
                    ('add_target_feature', Bowling_Single_Day()),
                    ('drop_target_feature', TargetBowling()),
                    ('select_features', ColumnTransformer([
                    ('keep_selected', 'passthrough', features['ODI'][2])  
                    ])),
                    ('regressor', xgb.XGBRegressor())
                ])
        }
    
    def odi_test_pipeline(self):
        return {
                0 : Pipeline([
                    ('add_target_feature', Batting_Test_Single())
                ]),
                1 : Pipeline([
                    ('add_target_feature', All_Rounder_Test_Single())
                ]),
                2 : Pipeline([
                    ('add_target_feature', Bowling_Test_Single())
                ])
        }
    def get_pipeline(self, format_type):
        return self.pipelines.get(format_type, None)