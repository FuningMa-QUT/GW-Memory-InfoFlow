# GW-Memory-InfoFlow
This repository contains the source code for the information-theoretic analysis presented in the manuscript: **"Nationwide information flows reveal long-term memory and asymmetric groundwater responses to recharge across Germany"** (Currently under review at *Hydrology and Earth System Sciences - HESS*).
# Groundwater Memory and Information Flow Analysis (Germany)

[![Paper Status](https://img.shields.io/badge/Status-Under_Review-yellow.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for the information-theoretic analysis presented in the manuscript:
**"Nationwide information flows reveal long-term memory and asymmetric groundwater responses to recharge across Germany"** (Currently under review at *Hydrology and Earth System Sciences - HESS*).

## 📌 Overview
This framework applies a model-free, discrete mutual information approach to quantify the seasonal coupling between effective recharge and groundwater levels across 3,207 monitoring wells in Germany (1991–2022). It extracts critical hydrologic signatures:
*   **Optimal Lag Time ($\tau_{opt}$)**
*   **Maximum Mutual Information ($I_{max}$)**
*   **Dry/Wet Asymmetry Ratio ($I_{s0}/I_{s1}$)**

## 📂 Repository Structure
*   `load_gems_data.py`: Preprocesses the raw GEMS-GER dataset, calculates the effective recharge proxy, normalizes groundwater levels, and builds a consolidated pickle file for fast I/O.
*   `compute_info_gems.py`: The core computational engine. It performs parallelized histogram scanning, evaluates mutual information, executes non-parametric surrogate significance testing, and calculates Kullback-Leibler divergences for state asymmetry. (Supports breakpoint resuming).
*   `final_thesis_summary.py`: Aggregates the results and generates the statistical summaries, seasonal breakdowns, and hydrogeological tabulations used in the manuscript (e.g., Table 1).

## 🚀 Usage
1. Ensure the required Python packages are installed: `pandas`, `numpy`, `scipy`, `joblib`.
2. Update the `PROJECT_DIR` paths in the scripts to match your local environment.
3. Run the pipeline in the following order:
   ```bash
   python load_gems_data.py
   python compute_info_gems.py
   python final_thesis_summary.py
