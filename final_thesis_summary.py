# final_thesis_summary.py - Comprehensive data summary output for the manuscript (3207 Wells)
import pandas as pd
import numpy as np
import os

# ====================== Path Settings ======================
PROJECT_DIR = r"D:\Desktop\GEMS_InfoFlow_Analysis"
RESULTS_CSV = os.path.join(PROJECT_DIR, r"results\info_results_all.csv")
STATIC_CSV  = os.path.join(PROJECT_DIR, r"data_static\gems_static.csv")

def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: Results file not found at {RESULTS_CSV}")
        return

    # 1. Load and merge data
    df_info = pd.read_csv(RESULTS_CSV)
    df_static = pd.read_csv(STATIC_CSV)
    df = df_info.merge(df_static, left_on='Well', right_on='MW_ID', how='left')
    
    # Preprocessing: Depth binning
    df['Depth_bin'] = pd.cut(df['Depth'], bins=[0, 15, 30, 60, 100, np.inf], 
                            labels=['Shallow(<15m)', 'Medium(15-30m)', 'Deep(30-60m)', 'VeryDeep(60-100m)', 'DeepAquifer(>100m)'])

    print("\n" + "="*80)
    print("      GERMANY GROUNDWATER INFO-FLOW ANALYSIS: FULL POPULATION (3207 WELLS)      ")
    print("="*80)

    # --- SECTION 1: Global Statistics (Abstract/Introduction) ---
    print("\n[SECTION 1: GLOBAL OVERVIEW]")
    total_wells = df['Well'].nunique()
    total_records = len(df)
    print(f"Total Wells Analyzed: {total_wells}")
    print(f"Total Significant Seasonal Samples: {total_records}")
    
    overall_tau = df['tau'].median()
    overall_ratio = df[df['ratio'] != np.inf]['ratio'].median()
    print(f"Overall Median Tau (Lag): {overall_tau:.2f} weeks")
    print(f"Overall Median Asymmetry Ratio: {overall_ratio:.3f}")

    # --- SECTION 2: Seasonal Characteristics (Results: Seasonal Dynamics) ---
    print("\n[SECTION 2: SEASONAL DYNAMICS]")
    seasonal = df.groupby('Season', observed=True).agg({
        'tau': ['median', lambda x: (x >= 26).mean() * 100],
        'I_max': 'median',
        'M': ['mean', 'median'],  
        'ratio': [lambda x: x[x != np.inf].median(), lambda x: (x == np.inf).mean() * 100]
    }).round(3)
    seasonal.columns = ['Tau_Median', 'Tau_LongLag_Pct(>=26w)', 'I_max_Median', 'M_Mean', 'M_Median', 'Ratio_Median', 'Ratio_Inf_Pct']
    print(seasonal)

    # --- SECTION 3: Hydrogeological Controls (Results: Hydrogeology - Tailored for Table 1) ---
    print("\n[SECTION 3: HYDROGEOLOGICAL CONTROLS (AQUIFER TYPE)]")
    aq_stats = df.groupby('AquiferMed').agg(
        Sample_Count=('tau', 'count'),
        Median_Tau_weeks=('tau', 'median'),
        Tau_LongLag_Pct=('tau', lambda x: (x >= 26).mean() * 100),
        Median_Ratio=('ratio', lambda x: x[x != np.inf].median()),
        Median_Imax=('I_max', 'median')
    )
    print(aq_stats.round(3))

    # --- SECTION 4: Depth Impact (Discussion: Depth Sensitivity) ---
    print("\n[SECTION 4: VERTICAL CHARACTERISTICS (DEPTH BINS)]")
    depth_stats = df.groupby('Depth_bin', observed=True).agg({
        'tau': 'median',
        'ratio': lambda x: x[x != np.inf].median(),
        'I_max': 'median'
    })
    print(depth_stats.round(3))

    # --- SECTION 5: Extreme Response Analysis (Discussion: Dry vs Wet) ---
    print("\n[SECTION 5: EXTREME RESPONSE ANALYSIS]")
    print("Percentage of wells where info is solely driven by 'Dry' precursors (Ratio=inf):")
    inf_summary = df.groupby('Season')['ratio'].apply(lambda x: (x == np.inf).mean() * 100).round(2)
    print(inf_summary)

    print("\n" + "="*80)
    print("END OF SUMMARY - PLEASE COPY THIS OUTPUT FOR ANALYSIS")
    print("="*80)

if __name__ == "__main__":
    main()
