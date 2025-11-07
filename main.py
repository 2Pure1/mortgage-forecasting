import pandas as pd
from src.data_processing import HMDAProcessor, main as process_data
from src.modeling import MortgageForecaster
from src.visualization import MortgageVisualizer
import yaml

def run_analysis():
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Process data
    print("=== Mortgage Origination Forecasting ===")
    print("Step 1: Data Processing")
    quarterly_data = process_data()
    
    # Initialize visualizer
    viz = MortgageVisualizer(config)
    
    # Exploratory plots
    print("\nStep 2: Exploratory Analysis")
    fig_ts, ax_ts = viz.plot_time_series(quarterly_data)
    fig_ts.savefig('outputs/time_series_plot.png', dpi=300, bbox_inches='tight')
    
    # Modeling
    print("\nStep 3: Model Building")
    forecaster = MortgageForecaster(quarterly_data)
    forecaster.prepare_data(test_size=0.2)
    
    # Fit models
    forecaster.fit_sarima()
    forecaster.fit_prophet()
    forecaster.fit_ets()
    
    # Generate forecasts
    horizon = config['model']['forecast_horizon']
    forecasts = forecaster.generate_forecasts(horizon=horizon)
    
    # Evaluate models
    print("\nStep 4: Model Evaluation")
    evaluation = forecaster.evaluate_models()
    print("\nModel Performance Comparison:")
    print(evaluation.round(2))
    
    # Plot forecast comparison
    fig_comp, ax_comp = viz.plot_forecast_comparison(
        forecaster.train, forecaster.test, forecasts, forecaster.models
    )
    fig_comp.savefig('outputs/forecast_comparison.png', dpi=300, bbox_inches='tight')
    
    # Final forecast
    print("\nStep 5: Final Forecast")
    final_forecast, best_model = forecaster.final_forecast(
        periods=config['forecasting']['final_forecast_quarters']
    )
    
    # Plot final forecast
    fig_final, ax_final = viz.plot_final_forecast(
        forecaster.data, final_forecast, best_model
    )
    fig_final.savefig('outputs/final_forecast.png', dpi=300, bbox_inches='tight')
    
    # Save results
    final_forecast.to_csv('outputs/final_forecast_results.csv', index=False)
    evaluation.to_csv('outputs/model_evaluation.csv')
    
    print(f"\n=== Analysis Complete ===")
    print(f"Best model: {best_model}")
    print(f"Forecast accuracy: {evaluation.loc[best_model, 'Accuracy']:.2f}%")
    print(f"Forecast horizon: {config['forecasting']['final_forecast_quarters']} quarters")
    print("Results saved to outputs/ directory")

if __name__ == "__main__":
    run_analysis()