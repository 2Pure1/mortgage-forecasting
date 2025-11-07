# Regional Mortgage Origination Volume Forecasting

## Project Overview
This project develops a time series forecasting model to predict mortgage origination volumes for specific geographic regions using HMDA (Home Mortgage Disclosure Act) data.

## Business Objective
Predict total dollar volume of mortgage originations for specific states or MSAs for the next 4-8 quarters to assist regional banks in strategic capital planning.

## Methodology
- **Data Source**: HMDA Loan Application Register (2018-2023)
- **Core Technique**: SARIMA time series forecasting
- **Comparison Models**: Facebook Prophet, Exponential Smoothing
- **Geography**: Configurable by MSA or state

## Key Features
- Automated data processing from raw HMDA files
- Multiple time series model comparison
- Comprehensive model evaluation metrics
- Production-ready forecasting pipeline
- Interactive visualization and reporting

## Project Structure
mortgage-forecasting/

├── data/

│   ├── raw/               # Raw HMDA files

│   ├── processed/         # Cleaned and aggregated data

│   └── external/          # BEA data (optional)

├── notebooks/

│   ├── 01_data_wrangling.ipynb

│   ├── 02_eda_analysis.ipynb

│   ├── 03_model_building.ipynb

│   └── 04_forecasting_evaluation.ipynb

├── src/

│   ├── data_processing.py

│   ├── modeling.py

│   └── visualization.py

├── config/

│   └── config.yaml

├── requirements.txt

├── README.md

└── main.py


## Quick Start
1. Clone repository and install dependencies:
```bash
pip install -r requirements.txt