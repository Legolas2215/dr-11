from model.train_model import *
from model.predict_model import *

def model_function(start_date_train, end_date_train, start_date_test, end_date_test):
    trainer = ModelTrainer()
    trainer.model_train(start_date_train, end_date_train)
    predictor = ModelPredictor()
    results = predictor.predict_model_ui(start_date_test, end_date_test)
    return results

def product_function(Format, Date, Team1, Team2, Player1_List, Player2_List):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_artifacts_path = os.path.join(base_dir, 'src', 'model_artifacts')
    print("Searching for model files...")
    model_files = {
        'bowler': os.path.join(model_artifacts_path, Format + '_bowler.pkl'),
        'batsman': os.path.join(model_artifacts_path, Format + '_batsman.pkl'),
        'allrounder': os.path.join(model_artifacts_path, Format + '_allrounder.pkl')
    }

    # Check if any of the model files are missing
    missing_models = [model_name for model_name, model_path in model_files.items() if not os.path.exists(model_path)]

    if missing_models:
        print(f"Missing models: {', '.join(missing_models)}. Training the models...")
        trainer = ModelTrainer()
        trainer.product_train(Format)
    predictor = ModelPredictor()
    result = predictor.predict_product_ui(Format, Date, Team1, Team2, Player1_List, Player2_List)
    return result
