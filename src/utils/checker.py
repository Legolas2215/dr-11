import os
from data_processing.feature_engineering import *
import pandas as pd
from data_processing.data_download import *

def check_raw():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    folder_path = os.path.join(base_dir, 'src', 'data', 'raw')
    print("Checking raw data...")

    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in the directory {folder_path}! Downloading raw data.")
        return False
    else:
        print("Raw data checked successfully! Processing interim data.")
        print("To Convert raw data to interim data, run the following command:")
        print("src/data_processing/data_download.py")
        return False

def check_interim():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    folder_path = os.path.join(base_dir, 'src', 'data', 'interim')
    print("Checking interim data...")

    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in the directory {folder_path}")
        check_raw()
    else:
        print("Interim data checked successfully!")
        print("To Convert interim data to final data, run the following command:")
        print("src/data_processing/data_download.py")
        return False

def check_processed():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    folder_path = os.path.join(base_dir, 'src', 'data', 'processed')
    print("Checking processed data...")

    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in the directory {folder_path}")
        check_interim()
    return True
    

def check_dates(start_date, end_date):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    folder_path = os.path.join(base_dir, 'src', 'data', 'processed')
    print("Checking processed data...")
    data_empty = True
    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df = prepare_datasets(df, start_date, end_date)

        if not df.empty:
            data_empty = False

    if data_empty:
        print(f"Error: No data found for the dates {start_date} to {end_date} in the directory {folder_path}")
        return False
    print("Processed data checked successfully!")
    return True