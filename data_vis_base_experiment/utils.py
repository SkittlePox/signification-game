import csv
import sys
import pandas as pd

def load_csv(filepath):
    """
    Load data from a CSV file with headers.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        list: List of dictionaries, where each dict represents a row
              with column headers as keys
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        # DictReader automatically uses first row as headers
        reader = csv.DictReader(file)
        
        # Convert to list to load all data into memory
        data = list(reader)
        
        # Get headers
        headers = reader.fieldnames
        
    return data, headers


def get_sweep_dirs(filepath):
    df = pd.read_csv(filepath)

    filtered_df = df[["Name", "ID", "LISTENER_ARCH", "SPEAKER_ARCH"]]


        

if __name__ == "__main__":
    load_csv("filename")
