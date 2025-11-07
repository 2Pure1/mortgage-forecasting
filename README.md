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
├── data/ # Raw and processed data
├── notebooks/ # Exploratory analysis Jupyter notebooks
├── src/ # Core Python modules
├── config/ # Configuration files
├── outputs/ # Results and visualizations
└── main.py # Main execution script


## Quick Start
1. Clone repository and install dependencies:
```bash
pip install -r requirements.txt
