# compute_info_gems.py - Parallel processing & breakpoint resume for full sample information flow analysis
import pickle
import numpy as np
from scipy.stats import entropy as scipy_entropy
import os
import pandas as pd
from joblib import Parallel, delayed
import ast 

# ====================== Paths & Parameters ======================
PROJECT_DIR = r'D:\Desktop\GEMS_InfoFlow_Analysis'
PICKLE_FILE = os.path.join(PROJECT_DIR, 'gems_processed_all.pkl')
RESULTS_CSV = os.path.join(PROJECT_DIR, 'results', 'info_results_all.csv')

# Automatically read the full sample well list
WELL_LIST_FILE = os.path.join(PROJECT_DIR, 'scripts', 'all_3207_wells.txt')

# Resource allocation: Recommended to use 70%-80% of total CPU cores to prevent system freeze.
# e.g., Set to 6 for an 8-core CPU, or 12 for a 16-core CPU.
N_CORES = 20  

# ====================== Core Computational Functions ======================
def compute_entropy(p, base=2):
    p = p[p > 0]
    return -np.sum(p * np.log(p) / np.log(base))

def compute_mutual_info_franzen(x, y, bins_y=5):
    valid = ~np.isnan(y) & ~np.isnan(x)
    x_valid = x[valid]
    y_valid = y[valid]
    if len(y_valid) < 20: return 0.0
    
    bins = np.percentile(y_valid, np.linspace(0, 100, bins_y + 1))
    y_binned = np.digitize(y_valid, bins) - 1
    joint_hist, _, _ = np.histogram2d(x_valid, y_binned, bins=[2, bins_y], range=[[0,1], [0, bins_y-1]])
    joint_p = joint_hist / joint_hist.sum()
    p_x, p_y = joint_p.sum(axis=1), joint_p.sum(axis=0)
    I = compute_entropy(p_x) + compute_entropy(p_y) - compute_entropy(joint_p.flatten())
    return max(I, 0.0)

def surrogate_test(I_obs, x, y, n_shuffle=50, bins_y=5):
    I_shuff = [compute_mutual_info_franzen(np.random.permutation(x), y, bins_y) for _ in range(n_shuffle)]
    I_crit = np.mean(I_shuff) + 3 * np.std(I_shuff)
    return I_obs > I_crit, I_crit

def get_season(month):
    if month in [12, 1, 2]: return 'DJF'
    elif month in [3, 4, 5]: return 'MAM'
    elif month in [6, 7, 8]: return 'JJA'
    else: return 'SON'

def compute_icrit_for_well(well_data, well_id, max_lag=52, n_thresholds=20, n_shuffle=50):
    if well_data is None: return None
    recharge = well_data['recharge_proxy']
    gwl_norm = well_data['gwl_norm']
    dates = pd.to_datetime(well_data['date'])
    
    valid = ~np.isnan(recharge) & ~np.isnan(gwl_norm)
    recharge, gwl_norm, dates_valid = recharge[valid], gwl_norm[valid], dates[valid]
    
    if len(recharge) < 200: return None
    
    thresholds = np.percentile(recharge, np.linspace(5, 95, n_thresholds))
    seasons = {'DJF': [], 'MAM': [], 'JJA': [], 'SON': []}
    for i, d in enumerate(dates_valid): seasons[get_season(d.month)].append(i)
    
    results = {}
    for season_name, idx_list in seasons.items():
        if len(idx_list) < 50: continue
        r_season, g_season = recharge[idx_list], gwl_norm[idx_list]
        I_max, best_tau, best_M, best_sig, best_crit = 0, 0, 0, False, 0
        
        for M in thresholds:
            binary_input = (r_season > M).astype(int)
            for tau in range(0, max_lag + 1):
                if tau >= len(binary_input): continue
                shifted = binary_input[:-tau] if tau > 0 else binary_input
                target = g_season[tau:]
                I = compute_mutual_info_franzen(shifted, target)
                if I > I_max:
                    is_sig, crit = surrogate_test(I, shifted, target, n_shuffle)
                    if is_sig:
                        I_max, best_tau, best_M, best_sig, best_crit = I, tau, M, is_sig, crit
        
        if best_sig:
            binary_best = (r_season > best_M).astype(int)
            p_full, _ = np.histogram(g_season, bins=10, density=True)
            p_dry, _ = np.histogram(g_season[binary_best==0], bins=10, density=True) if np.any(binary_best==0) else (p_full, None)
            p_wet, _ = np.histogram(g_season[binary_best==1], bins=10, density=True) if np.any(binary_best==1) else (p_full, None)
            
            I_s0, I_s1 = scipy_entropy(p_dry, p_full), scipy_entropy(p_wet, p_full)
            results[season_name] = {
                'I_max': I_max, 'tau': best_tau, 'M': best_M, 
                'I_s0': I_s0, 'I_s1': I_s1, 'ratio': I_s0/I_s1 if I_s1>0 else np.inf, 
                'significant': True
            }
    return results

# ====================== Main Execution Logic ======================
if __name__ == '__main__':
    # 1. Load well list
    with open(WELL_LIST_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        WELL_IDS = ast.literal_eval(content[content.find('['):content.rfind(']')+1])

    # 2. Breakpoint resume check (skip already computed wells)
    if os.path.exists(RESULTS_CSV):
        df_exist = pd.read_csv(RESULTS_CSV)
        done_wells = df_exist['Well'].unique().tolist()
        WELL_IDS = [wid for wid in WELL_IDS if wid not in done_wells]
        print(f"Skipped {len(done_wells)} already processed wells. Remaining: {len(WELL_IDS)}")
    
    if not WELL_IDS:
        print("All computations are completed!")
    else:
        # 3. Load data (keep only unprocessed wells in memory to prevent overflow)
        with open(PICKLE_FILE, 'rb') as f:
            all_data = pickle.load(f)['dynamic']
        to_process_data = {wid: all_data.get(wid) for wid in WELL_IDS if wid in all_data}
        del all_data # Free large memory block

        # 4. Parallel computation
        results_list = Parallel(n_jobs=N_CORES, verbose=10)(
            delayed(compute_icrit_for_well)(to_process_data.get(wid), wid) for wid in WELL_IDS
        )

        # 5. Organize and append results to CSV
        rows = []
        for wid, res_dict in zip(WELL_IDS, results_list):
            if res_dict:
                for s_name, res in res_dict.items():
                    rows.append({'Well': wid, 'Season': s_name, **res})
        
        if rows:
            df_new = pd.DataFrame(rows)
            write_header = not os.path.exists(RESULTS_CSV)
            df_new.to_csv(RESULTS_CSV, mode='a', index=False, header=write_header)
            print(f"Added {len(df_new)} new rows of results to {RESULTS_CSV}")

    print("Task finished.")
