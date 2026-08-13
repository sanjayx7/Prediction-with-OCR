import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# Set modern style for plots
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def preprocess_and_engineer(data_path):
    # Load data
    df = pd.read_csv(data_path)
    
    # Clean strings if any
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('"', '').str.strip()
            
    # Convert numeric columns
    df['Premium'] = df['Premium'].astype(float)
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    
    # Parse Date
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Feature Engineering
    # 1. Sin/Cos encoding of month (cyclical features)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    
    # 2. Year Fraction (representing continuous time progression)
    df['Year_Fraction'] = df['Year'] + (df['Month'] - 1) / 12.0
    
    # 3. Time elapsed in months from start of dataset
    start_date = df['Date'].min()
    df['Months_Elapsed'] = ((df['Date'].dt.year - start_date.year) * 12 + 
                            (df['Date'].dt.month - start_date.month))
    
    # Targets: We compute both percentage representations
    total_premium = df['Premium'].sum()
    max_premium = df['Premium'].max()
    
    df['Premium_Pct_Total'] = (df['Premium'] / total_premium) * 100
    df['Premium_Pct_Max'] = (df['Premium'] / max_premium) * 100
    
    return df

def train_and_evaluate(df, target_col, output_dir):
    # Chronological Split (Train: first 10 months, Test: last 3 months)
    # Since dataset is 13 rows, this is appropriate for time series forecasting
    train_size = 10
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    features = ['Year_Fraction', 'Month_Sin', 'Month_Cos', 'Months_Elapsed']
    
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_test = test_df[features].values
    y_test = test_df[target_col].values
    
    # Scale features (important for SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(df[features].values)
    
    # Define models
    # hyperparameters are kept simple to prevent overfitting on 13 samples
    models = {
        'SVR': SVR(C=10.0, epsilon=0.1, kernel='rbf'),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    print(f"\n--- Evaluation for Target: {target_col} ---")
    
    for name, model in models.items():
        # SVR uses scaled features, tree models use unscaled
        X_tr = X_train_scaled if name == 'SVR' else X_train
        X_te = X_test_scaled if name == 'SVR' else X_test
        X_al = X_all_scaled if name == 'SVR' else df[features].values
        
        # Fit model
        model.fit(X_tr, y_train)
        
        # Predict
        preds_train = model.predict(X_tr)
        preds_test = model.predict(X_te)
        preds_all = model.predict(X_al)
        
        # Evaluate
        mae = mean_absolute_error(y_test, preds_test)
        r2 = r2_score(y_test, preds_test)
        
        # Calculate train metrics for overfitting check
        mae_train = mean_absolute_error(y_train, preds_train)
        r2_train = r2_score(y_train, preds_train)
        
        print(f"{name}:")
        print(f"  Train -> MAE: {mae_train:.4f}, R²: {r2_train:.4f}")
        print(f"  Test  -> MAE: {mae:.4f}, R²: {r2:.4f}")
        
        results[name] = {
            'model': model,
            'mae_test': mae,
            'r2_test': r2,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'scaler': scaler if name == 'SVR' else None
        }
        predictions[name] = preds_all
        
    return results, predictions

def generate_plots(df, predictions_total, predictions_max, output_dir):
    # Plot 1: Percentage of Total Premium
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Premium_Pct_Total'], 'ko-', label='Actual Premium %', linewidth=2.5)
    
    colors = {'SVR': '#3b82f6', 'RandomForest': '#10b981', 'XGBoost': '#f59e0b'}
    for model_name, preds in predictions_total.items():
        plt.plot(df['Date'], preds, color=colors[model_name], linestyle='--', marker='x', 
                 label=f'{model_name} Predicted')
        
    # Mark the split point between train and test
    split_date = df['Date'].iloc[9]
    plt.axvline(x=split_date, color='red', linestyle=':', label='Train-Test Split')
    
    plt.title('Premium Prediction (Percentage of Total Premium Volume)', pad=20)
    plt.xlabel('Date')
    plt.ylabel('Premium Value (% of Total Sum)')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'premium_pct_total_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: Percentage of Max Premium
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Premium_Pct_Max'], 'ko-', label='Actual Premium %', linewidth=2.5)
    
    for model_name, preds in predictions_max.items():
        plt.plot(df['Date'], preds, color=colors[model_name], linestyle='--', marker='x', 
                 label=f'{model_name} Predicted')
        
    plt.axvline(x=split_date, color='red', linestyle=':', label='Train-Test Split')
    plt.title('Premium Prediction (Percentage of Max Monthly Premium)', pad=20)
    plt.xlabel('Date')
    plt.ylabel('Premium Value (% of Max Monthly)')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'premium_pct_max_comparison.png'), dpi=300)
    plt.close()
    
    # We also create a single premium-looking dashboard style plot showing the actual vs predicted values
    # for the best overall model (let's check which is best, usually SVR or RF on small datasets)
    # We will pick the SVR model for the final combined plot
    plt.figure(figsize=(10, 5))
    plt.scatter(df['Premium_Pct_Total'], predictions_total['SVR'], color='#3b82f6', s=80, alpha=0.8, 
                edgecolors='black', label='Predictions (SVR)')
    # Perfect fit line
    min_val = min(df['Premium_Pct_Total'].min(), predictions_total['SVR'].min())
    max_val = max(df['Premium_Pct_Total'].max(), predictions_total['SVR'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    plt.title('Actual vs. Predicted Premium Percentage (Total)', pad=15)
    plt.xlabel('Actual Percentage of Total')
    plt.ylabel('Predicted Percentage of Total')
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_fit.png'), dpi=300)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'premium_data.csv')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Section A: Premium Prediction Pipeline...")
    df = preprocess_and_engineer(data_path)
    
    # Save preprocessed dataset to output folder for checking
    df.to_csv(os.path.join(output_dir, 'premium_data_processed.csv'), index=False)
    print(f"Preprocessed dataset saved to output/premium_data_processed.csv")
    
    # Train and evaluate models on Target 1: Percentage of Total Premium
    results_total, predictions_total = train_and_evaluate(df, 'Premium_Pct_Total', output_dir)
    
    # Train and evaluate models on Target 2: Percentage of Max Premium
    results_max, predictions_max = train_and_evaluate(df, 'Premium_Pct_Max', output_dir)
    
    # Generate visualization plots
    generate_plots(df, predictions_total, predictions_max, output_dir)
    print("Plots generated and saved to the output/ directory.")
    
    # Save the models of the best fit (SVR and RandomForest) to disk
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save best models and scalers for both targets
    for name, res in results_total.items():
        joblib.dump(res['model'], os.path.join(models_dir, f'{name.lower()}_model_total.joblib'))
        if res['scaler'] is not None:
            joblib.dump(res['scaler'], os.path.join(models_dir, f'{name.lower()}_scaler_total.joblib'))
            
    for name, res in results_max.items():
        joblib.dump(res['model'], os.path.join(models_dir, f'{name.lower()}_model_max.joblib'))
        if res['scaler'] is not None:
            joblib.dump(res['scaler'], os.path.join(models_dir, f'{name.lower()}_scaler_max.joblib'))
            
    # Save a summary JSON with metrics
    metrics = {
        'total_premium': float(df['Premium'].sum()),
        'max_premium': float(df['Premium'].max()),
        'targets': {
            'Premium_Pct_Total': {
                model_name: {'mae': float(res['mae_test']), 'r2': float(res['r2_test'])}
                for model_name, res in results_total.items()
            },
            'Premium_Pct_Max': {
                model_name: {'mae': float(res['mae_test']), 'r2': float(res['r2_test'])}
                for model_name, res in results_max.items()
            }
        }
    }
    
    import json
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Model training metrics saved to output/model_metrics.json")
    print("Section A completed successfully!")

if __name__ == '__main__':
    main()
