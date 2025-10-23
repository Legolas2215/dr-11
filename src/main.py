import sys
import argparse
from utils.checker import *
from api.app import model_function, product_function

start_date_train = '2014-01-01'
start_date_test = '2022-01-20'
end_date_train = '2018-12-31'
end_date_test = '2022-01-28'

Format = "multi_day"
Team1 = "India"
Team2 = "Pakistan"
Player1_List = ["Virat Kohli", "Rohit Sharma", "Jasprit Bumrah", "Ravindra Jadeja", "Mohammed Shami", "KL Rahul", "Shikhar Dhawan", "Rishabh Pant", "Hardik Pandya", "Yuzvendra Chahal", "Shreyas Iyer", "Suryakumar Yadav"]
Player2_List = ["Babar Azam", "Shaheen Afridi", "Mohammad Rizwan", "Shadab Khan", "Fakhar Zaman", "Imad Wasim", "Haris Rauf", "Mohammad Hafeez", "Mohammad Hasnain", "Sarfaraz Ahmed", "Usman Qadir", "Asif Ali"] 

def main():
    parser = argparse.ArgumentParser(description='Process some flags.')
    parser.add_argument('--model', action='store_true', help='Run the model function')
    parser.add_argument('--product', action='store_true', help='Run the product function')
    args = parser.parse_args()

    if args.model:
        start_date_train = '2014-01-01'
        end_date_train = '2018-12-31'
        start_date_test = '2020-01-20'
        end_date_test = '2022-01-28'
        print("Starting model function...")
        if not (check_dates(start_date_train, end_date_train) | check_dates(start_date_test, end_date_test) | check_processed()):
            sys.exit(1)
        result = model_function(start_date_train, end_date_train, start_date_test, end_date_test)
        result.to_csv('results.csv', index=False)

    elif args.product:
        Format = "multi_day"
        Date = '2025-01-20'
        Team1 = "India"
        Team2 = "Pakistan"
        Player1_List = ["ba607b88", "740742ef", "462411b3", "fe93fd9d", "4a8a2e3b", "495d42a5", "0a476045", "919a3be2", "dbe50b21", "57ee1fde", "85ec8e33"]
        Player2_List = ["8a75e999", "fd2bf2a0", "2f26ac1a", "193ef196", "1777c020", "9cb8d7a6", "24bb1c2f", "64c34cd0", "29e253dd", "2254ab79", "b3118300"] 
        print("Starting product function...")
        result = product_function(Format, Date, Team1, Team2, Player1_List, Player2_List)
        result.to_csv('results.csv', index=False)
    else:
        print("Please provide either --model or --product flag.")
        sys.exit(1)

if __name__ == "__main__":
    main()