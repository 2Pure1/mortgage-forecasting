import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class HMDAProcessor:
    def __init__(self, config_path="config/model_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
    def load_hmda_data(self, years=None):
        """Load and combine HMDA data for multiple years"""
        if years is None:
            years = self.config['data']['hmda_years']
            
        all_data = []
        for year in years:
            file_path = Path(f"data/raw/hmda_{year}.csv")
            if file_path.exists():
                df = pd.read_csv(file_path, low_memory=False)
                # Select relevant columns
                cols = ['loan_amount', 'action_taken', 'state_code', 'county_code', 
                       'census_tract', 'msa_md', 'applicant_income_000s', 'loan_purpose',
                       'property_type', 'loan_type', 'as_of_year']
                
                # Only keep available columns
                available_cols = [col for col in cols if col in df.columns]
                df = df[available_cols]
                all_data.append(df)
                
        return pd.concat(all_data, ignore_index=True)
    
    def filter_originations(self, df):
        """Filter for originated loans (action_taken = 1)"""
        return df[df['action_taken'] == 1].copy()
    
    def aggregate_quarterly_volume(self, df, geography_type='msa'):
        """Aggregate loan volume by quarter for target geography"""
        # Convert to datetime (approximate quarter from year)
        df['quarter'] = pd.to_datetime(df['as_of_year'].astype(str) + 
                                     pd.Series(['-03-31', '-06-30', '-09-30', '-12-31'] * len(df)).sample(n=len(df), replace=True).values)
        
        # Filter for target geography
        geo_code = self.config['data']['geography_code']
        if geography_type == 'msa':
            df_geo = df[df['msa_md'] == geo_code]
        else:  # state
            df_geo = df[df['state_code'] == geo_code]
            
        # Aggregate by quarter
        quarterly_volume = df_geo.groupby('quarter')['loan_amount'].sum().reset_index()
        quarterly_volume.columns = ['date', 'total_loan_volume']
        quarterly_volume = quarterly_volume.sort_values('date')
        
        return quarterly_volume
    
    def create_time_series(self, df):
        """Create a complete time series with regular intervals"""
        # Ensure we have a regular time series
        start_date = df['date'].min()
        end_date = df['date'].max()
        full_dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        
        full_series = pd.DataFrame({'date': full_dates})
        full_series = full_series.merge(df, on='date', how='left')
        full_series['total_loan_volume'] = full_series['total_loan_volume'].fillna(0)
        
        return full_series

def main():
    processor = HMDAProcessor()
    
    # Load and process data
    print("Loading HMDA data...")
    hmda_data = processor.load_hmda_data()
    
    print("Filtering for originated loans...")
    originated_loans = processor.filter_originations(hmda_data)
    
    print("Aggregating quarterly volume...")
    quarterly_volume = processor.aggregate_quarterly_volume(originated_loans)
    
    print("Creating time series...")
    final_series = processor.create_time_series(quarterly_volume)
    
    # Save processed data
    output_path = Path("data/processed/quarterly_mortgage_volume.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_series.to_csv(output_path, index=False)
    
    print(f"Processed data saved to {output_path}")
    print(f"Time series range: {final_series['date'].min()} to {final_series['date'].max()}")
    print(f"Total quarters: {len(final_series)}")
    
    return final_series

if __name__ == "__main__":
    main()