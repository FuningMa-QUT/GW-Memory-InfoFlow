# load_gems_data.py - Automatically read WELL_IDS from txt and load dynamic/static data
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

# ====================== Project Paths ======================
PROJECT_DIR = r'D:\Desktop\GEMS_InfoFlow_Analysis'
STATIC_DIR = os.path.join(PROJECT_DIR, 'data_static')
DYNAMIC_DIR = os.path.join(PROJECT_DIR, 'data_dynamic')

# File paths
STATIC_FILE = os.path.join(STATIC_DIR, 'gems_static.csv')
WELL_LIST_FILE = os.path.join(PROJECT_DIR, 'scripts', 'all_3207_wells.txt')
OUTPUT_PICKLE = os.path.join(PROJECT_DIR, 'gems_processed_all.pkl')

def load_well_ids_from_file(file_path):
    """Read the WELL_IDS list from a txt file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Well list file not found: {file_path}\nPlease generate the txt file first.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Extract the WELL_IDS = [...] section
    if 'WELL_IDS =' not in content:
        raise ValueError("Incorrect file format: 'WELL_IDS =' line not found.")
    
    # Parse the list assuming standard Python list format
    start = content.find('[')
    end = content.rfind(']') + 1
    list_str = content[start:end]
    
    # Safely evaluate the string as a Python list
    import ast
    try:
        well_ids = ast.literal_eval(list_str)
        if not isinstance(well_ids, list):
            raise ValueError("Parsed result is not a list.")
        print(f"Successfully loaded {len(well_ids)} well IDs from file.")
        return well_ids
    except Exception as e:
        raise ValueError(f"Failed to parse WELL_IDS: {e}\nPlease check if the file format is a standard Python list.")

def load_static():
    """Read the static table containing well metadata"""
    df_static = pd.read_csv(STATIC_FILE)
    print("Total rows in static table:", len(df_static))
    return df_static

def load_dynamic(well_id):
    """Read the dynamic CSV time series for a single well"""
    file_path = os.path.join(DYNAMIC_DIR, f'{well_id}.csv')
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping {well_id}")
        return None
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # The date column is typically the first column ('Unnamed: 0')
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    try:
        gwl = df['GWL'].values
        precip = df['HYRAS_pr'].values
        pet = df['DWD_evapo_p'].values
    except KeyError as e:
        print(f"{well_id} is missing key columns: {e}")
        return None
    
    # Calculate effective recharge proxy
    recharge_proxy = np.maximum(precip - pet, 0)
    
    # Normalize Groundwater Levels (z-score)
    gwl_mean = np.nanmean(gwl)
    gwl_std = np.nanstd(gwl)
    gwl_norm = (gwl - gwl_mean) / gwl_std if gwl_std > 0 else gwl
    
    print(f"{well_id} data loaded successfully: {len(gwl)} records.")
    
    return {
        'date': df[date_col].values,
        'gwl': gwl,
        'gwl_norm': gwl_norm,
        'precip': precip,
        'pet': pet,
        'recharge_proxy': recharge_proxy
    }

def main():
    # 1. Automatically load WELL_IDS
    try:
        WELL_IDS = load_well_ids_from_file(WELL_LIST_FILE)
    except Exception as e:
        print(f"Failed to load WELL_IDS: {e}")
        return
    
    # 2. Read static metadata
    static_df = load_static()
    
    # 3. Load dynamic time series data
    data_dict = {}
    loaded_count = 0
    for wid in WELL_IDS:
        data = load_dynamic(wid)
        if data is not None:
            data_dict[wid] = data
            loaded_count += 1
    
    print(f"\nRequested to load {len(WELL_IDS)} wells; Successfully loaded {loaded_count} wells.")
    
    # 4. Save processed data to pickle
    processed = {
        'static': static_df,
        'dynamic': data_dict
    }
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(processed, f)
    print(f"Data preprocessing completed. Saved to: {OUTPUT_PICKLE}")

if __name__ == '__main__':
    main()
