# Floods_ML

This repository contains the source code and virtual prototype for a Decision Support System (DSS) developed as part of a Team Project at Lodz University of Technology (2026). The pipeline predicts flood occurrences in Poland using historical meteorological data.

## Project Overview
Instead of relying on computationally heavy 3D physical models, this project uses a data-driven approach. By engineering targeted features—most notably a rolling 3-month precipitation sum to act as a mathematical proxy for soil saturation (catchment memory) — we trained a **Random Forest classifier** to isolate rare extreme weather events (1% class prevalence).

## Features
* **Automated Data Cleaning:** Handles Polish diacritics and resolves IMGW-specific measurement status codes.
* **Context-Aware Feature Engineering:** Calculates cyclical time variables (sine/cosine), precipitation lags, and snowmelt risk interactions.
* **Strict Cross-Validation:** Uses `GroupShuffleSplit` by Station ID to ensure spatial validation without data leakage.
* **Imbalance-Aware Evaluation:** Optimizes and evaluates models using PR-AUC (Precision-Recall Area Under the Curve) rather than standard ROC/Accuracy.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/Team-Project-2026-Team5/Floods_ML.git](https://github.com/Team-Project-2026-Team5/Floods_ML.git)
   cd Floods_ML
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the pipeline:
   ```bash
   python flood_prediction_model.py

## Data Source
The primary dataset is based on the official monthly precipitation database from the Institute of Meteorology and Water Management (IMGW-PIB), covering the periods 1996–2001 and 2010.

## Authors
1. A. Nikohosian
2. A. Pokotylov
3. M. Świderek
4. M. Woźniak
