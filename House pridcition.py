"""
California Housing Price Prediction
====================================
A complete machine learning pipeline for predicting house prices using the California housing dataset.
Includes data cleaning, feature engineering, model training, and comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class HousingPricePredictor:
    """
    A comprehensive housing price prediction system with data preprocessing,
    feature engineering, and multiple model training capabilities.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the California housing dataset"""
        print("=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)
        
        # Load from local CSV file
        self.data = pd.read_csv('/home/claude/california_housing.csv')
        feature_names = [col for col in self.data.columns if col != 'MedHouseVal']
        
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"\nFeatures: {feature_names}")
        print(f"Target: MedHouseVal (Median House Value in $100,000s)")
        
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA EXPLORATION")
        print("=" * 80)
        
        # Basic statistics
        print("\n--- Dataset Info ---")
        print(self.data.info())
        
        print("\n--- Statistical Summary ---")
        print(self.data.describe())
        
        # Check for missing values
        print("\n--- Missing Values ---")
        missing = self.data.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        print(f"\n--- Duplicates ---")
        print(f"Number of duplicate rows: {duplicates}")
        
        return self.data.describe()
    
    def clean_data(self):
        """Clean the dataset"""
        print("\n" + "=" * 80)
        print("STEP 3: DATA CLEANING")
        print("=" * 80)
        
        initial_shape = self.data.shape
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        print(f"\nRemoved {initial_shape[0] - self.data.shape[0]} duplicate rows")
        
        # Handle missing values (if any)
        if self.data.isnull().sum().sum() > 0:
            self.data = self.data.fillna(self.data.median())
            print("Missing values filled with median")
        
        # Remove outliers using IQR method
        print("\n--- Removing Outliers ---")
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        mask = ~((self.data < lower_bound) | (self.data > upper_bound)).any(axis=1)
        self.data = self.data[mask]
        
        print(f"Removed {initial_shape[0] - self.data.shape[0]} outlier rows")
        print(f"Final dataset shape: {self.data.shape}")
        
        return self.data
    
    def feature_engineering(self):
        """Create new features from existing ones"""
        print("\n" + "=" * 80)
        print("STEP 4: FEATURE ENGINEERING")
        print("=" * 80)
        
        # Create new features
        self.data['RoomsPerHousehold'] = self.data['AveRooms'] / self.data['AveOccup']
        self.data['BedroomsPerRoom'] = self.data['AveBedrms'] / self.data['AveRooms']
        self.data['PopulationPerHousehold'] = self.data['Population'] / self.data['HouseAge']
        
        print("\nNew features created:")
        print("  - RoomsPerHousehold: Average rooms divided by average occupancy")
        print("  - BedroomsPerRoom: Ratio of bedrooms to total rooms")
        print("  - PopulationPerHousehold: Population density metric")
        
        print(f"\nUpdated dataset shape: {self.data.shape}")
        
        return self.data
    
    def prepare_features(self):
        """Prepare features and target for modeling"""
        print("\n" + "=" * 80)
        print("STEP 5: FEATURE PREPARATION")
        print("=" * 80)
        
        # Separate features and target
        X = self.data.drop('MedHouseVal', axis=1)
        y = self.data['MedHouseVal']
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\nFeatures scaled using StandardScaler")
        
        # Display feature importance (correlation with target)
        print("\n--- Feature Correlation with Target ---")
        correlations = pd.DataFrame({
            'Feature': X.columns,
            'Correlation': [self.data[col].corr(y) for col in X.columns]
        }).sort_values('Correlation', ascending=False)
        print(correlations.to_string(index=False))
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple regression models"""
        print("\n" + "=" * 80)
        print("STEP 6: MODEL TRAINING")
        print("=" * 80)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='r2')
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_test_pred
            }
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.models, self.results
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 80)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 80)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test R²': [self.results[m]['test_r2'] for m in self.results],
            'Test RMSE': [self.results[m]['test_rmse'] for m in self.results],
            'Test MAE': [self.results[m]['test_mae'] for m in self.results],
            'CV R² Mean': [self.results[m]['cv_mean'] for m in self.results],
            'CV R² Std': [self.results[m]['cv_std'] for m in self.results]
        }).sort_values('Test R²', ascending=False)
        
        print("\n--- Model Comparison ---")
        print(results_df.to_string(index=False))
        
        # Identify best model
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test R² Score: {results_df.iloc[0]['Test R²']:.4f}")
        print(f"   Test RMSE: {results_df.iloc[0]['Test RMSE']:.4f}")
        
        return results_df
    
    def visualize_results(self):
        """Create visualizations of the results"""
        print("\n" + "=" * 80)
        print("STEP 8: VISUALIZATION")
        print("=" * 80)
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Model Comparison - R² Scores
        ax1 = plt.subplot(2, 3, 1)
        models_list = list(self.results.keys())
        r2_scores = [self.results[m]['test_r2'] for m in models_list]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models_list)))
        bars = ax1.barh(models_list, r2_scores, color=colors)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Model Comparison - R² Score', fontweight='bold')
        ax1.set_xlim([0, 1])
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            ax1.text(score + 0.01, i, f'{score:.4f}', va='center')
        
        # 2. RMSE Comparison
        ax2 = plt.subplot(2, 3, 2)
        rmse_scores = [self.results[m]['test_rmse'] for m in models_list]
        bars = ax2.barh(models_list, rmse_scores, color=colors)
        ax2.set_xlabel('RMSE')
        ax2.set_title('Model Comparison - RMSE', fontweight='bold')
        for i, (bar, score) in enumerate(zip(bars, rmse_scores)):
            ax2.text(score + 0.01, i, f'{score:.4f}', va='center')
        
        # 3. Actual vs Predicted (Best Model)
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        best_name = best_model[0]
        best_predictions = best_model[1]['predictions']
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(self.y_test, best_predictions, alpha=0.5, s=10)
        ax3.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Price ($100k)')
        ax3.set_ylabel('Predicted Price ($100k)')
        ax3.set_title(f'Actual vs Predicted - {best_name}', fontweight='bold')
        ax3.legend()
        
        # 4. Residuals Plot
        ax4 = plt.subplot(2, 3, 4)
        residuals = self.y_test - best_predictions
        ax4.scatter(best_predictions, residuals, alpha=0.5, s=10)
        ax4.axhline(y=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Predicted Price ($100k)')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residual Plot', fontweight='bold')
        
        # 5. Feature Importance (for Random Forest)
        if 'Random Forest' in self.models:
            ax5 = plt.subplot(2, 3, 5)
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': self.data.drop('MedHouseVal', axis=1).columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            ax5.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax5.set_xlabel('Importance')
            ax5.set_title('Feature Importance - Random Forest', fontweight='bold')
        
        # 6. Prediction Error Distribution
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax6.axvline(x=0, color='r', linestyle='--', lw=2)
        ax6.set_xlabel('Prediction Error ($100k)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Prediction Errors', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/housing_price_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved: housing_price_analysis.png")
        
        return fig
    
    def run_pipeline(self):
        """Execute the complete ML pipeline"""
        print("\n" + "█" * 80)
        print("CALIFORNIA HOUSING PRICE PREDICTION - MACHINE LEARNING PIPELINE")
        print("█" * 80)
        
        # Execute all steps
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.feature_engineering()
        self.prepare_features()
        self.train_models()
        results_df = self.evaluate_models()
        self.visualize_results()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print(f"  • Dataset: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        print(f"  • Best Model: {results_df.iloc[0]['Model']}")
        print(f"  • Best R² Score: {results_df.iloc[0]['Test R²']:.4f}")
        print(f"  • Best RMSE: ${results_df.iloc[0]['Test RMSE'] * 100:.2f}k")
        print("\nOutputs:")
        print("  ✓ Visualizations saved to housing_price_analysis.png")
        print("  ✓ Models trained and evaluated")
        print("  ✓ Results available in the predictor object")
        
        return self


# Main execution
if __name__ == "__main__":
    # Create and run the predictor
    predictor = HousingPricePredictor()
    predictor.run_pipeline()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': list(predictor.results.keys()),
        'Test_R2': [predictor.results[m]['test_r2'] for m in predictor.results],
        'Test_RMSE': [predictor.results[m]['test_rmse'] for m in predictor.results],
        'Test_MAE': [predictor.results[m]['test_mae'] for m in predictor.results],
        'CV_R2_Mean': [predictor.results[m]['cv_mean'] for m in predictor.results],
        'CV_R2_Std': [predictor.results[m]['cv_std'] for m in predictor.results]
    }).sort_values('Test_R2', ascending=False)
    
    results_df.to_csv('/mnt/user-data/outputs/model_results.csv', index=False)
    print("\n📊 Results exported to model_results.csv")