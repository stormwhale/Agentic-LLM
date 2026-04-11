from smc_0_4 import get_smc_response, MarketData, get_deepseek_smc_response
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
import json
from openai import OpenAI
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

#Drag and drop the json file into the terminal:
def on_drop(event):
    file_paths = root.tk.splitlist(event.data)
    for path in file_paths:
        print(f'Dropped file: {path}')
        listbox.insert(tk.END, path)

# Load the API key from the local environment file
load_dotenv("vantage_key.env")
vantage_key = os.getenv("vantage_key", "")

load_dotenv("TD_api_key.env")
TD_api_key = os.getenv("TD_api_key", "")

load_dotenv("deepseek_key.env")
deepseek_key = os.getenv("deepseek_key", "")

#Initialize the market data object:
market_data = MarketData(vantage_api_key=vantage_key, TD_api_key=TD_api_key)

#====================================
#Change the test_data file over here:
#====================================
df_screened = pd.read_json('generated_questions/deepeval_questions3.json')

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

#Create a folder to store the results:
os.makedirs("generated_answers", exist_ok=True)

#Create a switch to flip between Qwen and DeepSeek:
model_selection = input('Which model would you like to use? (qwen or DeepSeek or quit to exit)?')
if model_selection == "qwen":
    print("\nRunning Qwen SMC to populate response + retrieved_contexts...")
    for e_num, (i, row) in enumerate(df_screened.iterrows(), start=1):
        print(f"  [{e_num}/{len(df_screened)}] {row['user_input'][:70]}...")
        result = get_smc_response(row["user_input"], market_data)
        df_screened.at[i, "response"] = result["response"]
        df_screened.at[i, "retrieved_contexts"] = result["retrieved_contexts"]
        df_screened.at[i, "actual_tool_calls"] = result["actual_tool_calls"]

    #Save the new dataframe to json:
    df_screened.to_json(f'generated_answers/test_question_filled_{model_selection}_{timestamp}.json', orient='records', indent=2)

elif model_selection == "DeepSeek":
    print("\nRunning DeepSeek SMC to populate response + retrieved_contexts...")
    for e_num, (i, row) in enumerate(df_screened.iterrows(), start=1):
        print(f"  [{e_num}/{len(df_screened)}] {row['user_input'][:70]}...")
        result = get_deepseek_smc_response(row["user_input"], market_data, deepseek_key)
        df_screened.at[i, "response"] = result["response"]
        df_screened.at[i, "retrieved_contexts"] = result["retrieved_contexts"]
        df_screened.at[i, "actual_tool_calls"] = result["actual_tool_calls"]

    #Save the new dataframe to json:
    df_screened.to_json(f'./generated_answers/test_question_filled_{model_selection}_{timestamp}.json', orient='records', indent=2)


