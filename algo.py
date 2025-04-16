import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import requests
import io
import os
import warnings
warnings.filterwarnings('ignore')

class ClimateRiskModel:
    def __init__(self):
        self.raw_data = {}
        self.processed_data = None
        self.combined_data = None
        self.model = None
        self.risk_scores = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.score_scaler = MinMaxScaler(feature_range=(1, 100))
        
    def fetch_data(self):
        """
        Fetch data from various sources and store in raw_data dictionary.
        In a real application, this would connect to APIs or download files.
        Here we're simulating with sample data.
        """
        print("Fetching climate and socioeconomic data for all U.S. states...")
        
        # Create simulated data for all states
        states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
            'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
            'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
            'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
            'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 
            'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 
            'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
        ]
        
        # 1. Temperature data (historical and projected)
        temp_data = pd.DataFrame({
            'state': states,
            'avg_temp_historical': np.random.normal(15, 5, 50),  # Celsius
            'temp_increase_projection': np.random.uniform(1.5, 4.5, 50),  # Celsius increase by 2050
            'heatwave_days_historical': np.random.randint(5, 35, 50),
            'heatwave_days_projection': np.random.randint(20, 80, 50),
        })
        self.raw_data['temperature'] = temp_data
        
        # 2. Precipitation and drought data
        precip_data = pd.DataFrame({
            'state': states,
            'avg_annual_precip_historical': np.random.normal(1000, 300, 50),  # mm
            'precip_change_projection': np.random.normal(0, 15, 50),  # percent change
            'drought_frequency_historical': np.random.uniform(0.1, 0.4, 50),  # frequency of drought years
            'drought_severity_projection': np.random.uniform(1.0, 2.5, 50),  # multiplier for future severity
        })
        self.raw_data['precipitation'] = precip_data
        
        # 3. Sea level and coastal data (coastal states have higher values)
        coastal_states = ['Alabama', 'Alaska', 'California', 'Connecticut', 'Delaware', 'Florida', 
                        'Georgia', 'Hawaii', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 
                        'Mississippi', 'New Hampshire', 'New Jersey', 'New York', 'North Carolina', 
                        'Oregon', 'Rhode Island', 'South Carolina', 'Texas', 'Virginia', 'Washington']
        
        coastal_data = pd.DataFrame({
            'state': states,
            'coastline_length': [np.random.uniform(500, 2000) if state in coastal_states else 0 for state in states],
            'sea_level_rise_projection': [np.random.uniform(0.3, 0.9) if state in coastal_states else 0 for state in states],  # meters by 2050
            'coastal_property_value': [np.random.uniform(10, 200) if state in coastal_states else 0 for state in states],  # billions USD
            'population_in_flood_zone': [np.random.uniform(5, 30) if state in coastal_states else 0 for state in states],  # percentage
        })
        self.raw_data['coastal'] = coastal_data
        
        # 4. Economic data and GDP sectors
        economic_data = pd.DataFrame({
            'state': states,
            'gdp_per_capita': np.random.uniform(40000, 80000, 50),  # USD
            'agriculture_pct_gdp': np.random.uniform(1, 15, 50),  # percentage
            'tourism_pct_gdp': np.random.uniform(2, 20, 50),  # percentage
            'outdoor_recreation_pct_gdp': np.random.uniform(1, 10, 50),  # percentage
            'climate_sensitive_industries_pct': np.random.uniform(10, 40, 50),  # percentage
        })
        self.raw_data['economic'] = economic_data
        
        # 5. Natural disaster data
        disaster_data = pd.DataFrame({
            'state': states,
            'wildfire_risk': np.random.uniform(0, 1, 50),  # normalized risk
            'flood_risk': np.random.uniform(0, 1, 50),  # normalized risk
            'hurricane_risk': [np.random.uniform(0.5, 1) if state in ['Florida', 'Louisiana', 'Texas', 'North Carolina', 
                                                                    'South Carolina', 'Alabama', 'Mississippi', 'Georgia'] 
                            else np.random.uniform(0, 0.3) for state in states],  # higher for hurricane-prone states
            'disaster_damages_historical': np.random.uniform(0.5, 10, 50),  # billions USD average annual
        })
        self.raw_data['disasters'] = disaster_data
        
        # 6. Infrastructure resilience
        infrastructure_data = pd.DataFrame({
            'state': states,
            'infrastructure_age': np.random.uniform(20, 60, 50),  # average age in years
            'resilience_investment': np.random.uniform(10, 100, 50),  # dollars per capita
            'grid_reliability': np.random.uniform(0.5, 0.99, 50),  # reliability score
            'water_infrastructure_vulnerability': np.random.uniform(0.1, 0.8, 50),  # vulnerability score
        })
        self.raw_data['infrastructure'] = infrastructure_data
        
        # 7. Social vulnerability
        social_data = pd.DataFrame({
            'state': states,
            'poverty_rate': np.random.uniform(8, 25, 50),  # percentage
            'uninsured_rate': np.random.uniform(5, 20, 50),  # percentage
            'elderly_population': np.random.uniform(10, 25, 50),  # percentage
            'socioeconomic_vulnerability_index': np.random.uniform(0.2, 0.8, 50),  # index 0-1
        })
        self.raw_data['social'] = social_data
        
        # 8. Adaptation and mitigation efforts
        adaptation_data = pd.DataFrame({
            'state': states,
            'climate_policy_strength': np.random.uniform(0, 1, 50),  # policy strength score
            'renewable_energy_pct': np.random.uniform(5, 45, 50),  # percentage of energy
            'adaptation_budget': np.random.uniform(1, 50, 50),  # dollars per capita
            'emission_reduction_targets': np.random.uniform(0, 50, 50),  # percentage reduction targeted
        })
        self.raw_data['adaptation'] = adaptation_data
        
        # 9. Ecological sensitivity
        ecological_data = pd.DataFrame({
            'state': states,
            'biodiversity_risk': np.random.uniform(0.1, 0.9, 50),  # risk score
            'forest_cover_change': np.random.normal(-5, 10, 50),  # percentage change projected
            'freshwater_stress_projection': np.random.uniform(0.1, 0.9, 50),  # stress score
            'ecosystem_services_value': np.random.uniform(1, 10, 50),  # billions USD
        })
        self.raw_data['ecological'] = ecological_data
        
        print("Data fetching completed.")
        return self.raw_data
        
    def preprocess_data(self):
        """
        Preprocess and combine all data sources into a single dataframe.
        """
        print("Preprocessing and combining datasets...")
        
        # If raw data is empty, fetch it first
        if not self.raw_data:
            self.fetch_data()
        
        # Combine all datasets on state column
        combined_df = self.raw_data['temperature']
        
        for key, df in self.raw_data.items():
            if key != 'temperature':
                combined_df = pd.merge(combined_df, df, on='state')
        
        # Set state as index
        combined_df.set_index('state', inplace=True)
        
        # Handle any missing values (in real-world data this would be more complex)
        combined_df.fillna(combined_df.mean(), inplace=True)
        
        self.combined_data = combined_df
        print(f"Combined dataset created with {combined_df.shape[1]} features for {combined_df.shape[0]} states.")
        
        return self.combined_data
    
    def engineer_features(self):
        """
        Feature engineering: create new features and prepare data for modeling.
        """
        print("Engineering features for climate risk assessment...")
        
        if self.combined_data is None:
            self.preprocess_data()
        
        df = self.combined_data.copy()
        
        # Create compound features that reflect interactions between factors
        
        # Temperature stress index
        df['temperature_stress_index'] = (
            df['temp_increase_projection'] * 
            df['heatwave_days_projection'] / df['heatwave_days_historical']
        )
        
        # Water stress index
        df['water_stress_index'] = (
            df['drought_severity_projection'] * 
            (1 + abs(df['precip_change_projection'])/100) *
            (1 / (df['avg_annual_precip_historical'] / 1000))  # Normalize by precipitation
        )
        
        # Coastal vulnerability index (higher for coastal states with more exposure)
        df['coastal_vulnerability_index'] = (
            df['coastline_length'] * 
            df['sea_level_rise_projection'] * 
            df['population_in_flood_zone'] / 100
        )
        
        # Economic sensitivity index
        df['economic_sensitivity_index'] = (
            df['agriculture_pct_gdp'] + 
            df['tourism_pct_gdp'] + 
            df['outdoor_recreation_pct_gdp'] +
            df['climate_sensitive_industries_pct']
        ) / 100  # Convert to proportion
        
        # Disaster risk composite
        df['disaster_risk_composite'] = (
            df['wildfire_risk'] + 
            df['flood_risk'] + 
            df['hurricane_risk']
        ) / 3
        
        # Infrastructure vulnerability
        df['infrastructure_vulnerability'] = (
            df['infrastructure_age'] / 50 +  # Normalize age
            (1 - df['grid_reliability']) +
            df['water_infrastructure_vulnerability'] -
            df['resilience_investment'] / 100  # Normalize investment
        )
        
        # Social vulnerability composite
        df['social_vulnerability_composite'] = (
            df['poverty_rate'] / 100 +
            df['uninsured_rate'] / 100 +
            df['elderly_population'] / 100 +
            df['socioeconomic_vulnerability_index']
        ) / 4
        
        # Adaptation capacity (inverse of vulnerability)
        df['adaptation_capacity'] = (
            df['climate_policy_strength'] +
            df['renewable_energy_pct'] / 100 +
            df['adaptation_budget'] / 50 +
            df['emission_reduction_targets'] / 100
        ) / 4
        
        # Ecological vulnerability
        df['ecological_vulnerability'] = (
            df['biodiversity_risk'] +
            (df['forest_cover_change'] < 0).astype(int) * abs(df['forest_cover_change']) / 20 +
            df['freshwater_stress_projection'] -
            df['ecosystem_services_value'] / 10
        ) / 4
        
        # Calculate an initial risk score as a weighted sum of our engineered features
        # This will be our target variable for the machine learning model
        df['initial_risk_score'] = (
            df['temperature_stress_index'] * 0.15 +
            df['water_stress_index'] * 0.15 +
            df['coastal_vulnerability_index'] * 0.15 +
            df['economic_sensitivity_index'] * 0.1 +
            df['disaster_risk_composite'] * 0.15 +
            df['infrastructure_vulnerability'] * 0.1 +
            df['social_vulnerability_composite'] * 0.1 -
            df['adaptation_capacity'] * 0.1 +
            df['ecological_vulnerability'] * 0.1
        )
        
        # Normalize the initial risk score to have a reasonable range
        min_score = df['initial_risk_score'].min()
        max_score = df['initial_risk_score'].max()
        df['initial_risk_score'] = (df['initial_risk_score'] - min_score) / (max_score - min_score) * 100
        
        # Store processed data
        self.processed_data = df
        
        # Define features and target
        self.target = df['initial_risk_score']
        self.features = df.drop('initial_risk_score', axis=1)
        
        print(f"Feature engineering complete. Created {len(df.columns) - len(self.combined_data.columns)} new features.")
        
        return self.processed_data
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train a model to predict climate risk scores.
        """
        print("Training climate risk assessment model...")
        
        if self.processed_data is None:
            self.engineer_features()
        
        # Extract features and target
        X = self.features
        y = self.target
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
        
        # Initialize and train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained. Test MSE: {mse:.2f}, R²: {r2:.2f}")
        
        # Get feature importances
        feature_importances = pd.Series(
            self.model.feature_importances_,
            index=self.features.columns
        ).sort_values(ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importances.head(10))
        
        return self.model
    
    def generate_risk_scores(self):
        """
        Generate comprehensive risk scores for all states.
        """
        print("Generating final climate risk scores for all states...")
        
        if self.model is None:
            self.train_model()
        
        # Get feature data
        X = self.features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        predicted_scores = self.model.predict(X_scaled)
        
        # Scale scores to 1-100 range for easier interpretation
        scaled_scores = self.score_scaler.fit_transform(predicted_scores.reshape(-1, 1)).flatten()
        
        # Create a dataframe with states and their risk scores
        risk_df = pd.DataFrame({
            'state': self.processed_data.index,
            'risk_score': scaled_scores
        })
        
        # Rank states by risk score
        risk_df['rank'] = risk_df['risk_score'].rank(ascending=False).astype(int)
        risk_df = risk_df.sort_values('rank')
        
        # Add risk categories
        risk_df['risk_category'] = pd.cut(
            risk_df['risk_score'], 
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )
        
        self.risk_scores = risk_df
        print("Risk scoring complete.")
        
        return self.risk_scores
    
    def analyze_risk_factors(self, state_name):
        """
        Analyze key risk factors for a specific state.
        """
        if self.risk_scores is None:
            self.generate_risk_scores()
            
        if state_name not in self.processed_data.index:
            print(f"State '{state_name}' not found in dataset.")
            return None
        
        # Get state data
        state_data = self.processed_data.loc[state_name]
        
        # Get engineered features for this state
        key_factors = [
            'temperature_stress_index', 'water_stress_index', 'coastal_vulnerability_index',
            'economic_sensitivity_index', 'disaster_risk_composite', 'infrastructure_vulnerability',
            'social_vulnerability_composite', 'adaptation_capacity', 'ecological_vulnerability'
        ]
        
        # Create a dataframe of key factors
        factors_df = pd.DataFrame({
            'factor': key_factors,
            'value': [state_data[factor] for factor in key_factors]
        })
        
        # Normalize values for comparison
        all_states_factors = self.processed_data[key_factors]
        min_vals = all_states_factors.min()
        max_vals = all_states_factors.max()
        
        factors_df['normalized_value'] = [
            (state_data[factor] - min_vals[factor]) / (max_vals[factor] - min_vals[factor])
            for factor in key_factors
        ]
        
        # Get state rank and score
        state_risk = self.risk_scores[self.risk_scores['state'] == state_name].iloc[0]
        
        print(f"\nRisk Assessment for {state_name}:")
        print(f"Overall Risk Score: {state_risk['risk_score']:.1f}/100 (Rank: {state_risk['rank']}/{len(self.processed_data)} states)")
        print(f"Risk Category: {state_risk['risk_category']}")
        
        print("\nKey Risk Factors (normalized 0-1 scale, higher = more vulnerable):")
        # Sort factors by normalized value in descending order
        factors_df = factors_df.sort_values('normalized_value', ascending=False)
        for _, row in factors_df.iterrows():
            if row['factor'] == 'adaptation_capacity':
                # For adaptation capacity, lower is worse
                print(f"  {row['factor']}: {1-row['normalized_value']:.2f} (higher is better)")
            else:
                print(f"  {row['factor']}: {row['normalized_value']:.2f}")
        
        return factors_df
    
    def visualize_results(self):
        """
        Create visualizations of the risk assessment results.
        """
        if self.risk_scores is None:
            self.generate_risk_scores()
        
        # Set up plotting
        plt.figure(figsize=(15, 10))
        sns.set_style("whitegrid")
        
        # 1. Bar chart of top 10 and bottom 10 states by risk score
        plt.subplot(2, 1, 1)
        top_states = self.risk_scores.nsmallest(10, 'rank')
        bottom_states = self.risk_scores.nlargest(10, 'rank')
        combined = pd.concat([top_states, bottom_states])
        
        bars = sns.barplot(x='state', y='risk_score', data=combined, 
                  palette=['red' if x <= 10 else 'green' for x in combined['rank']])
        
        plt.xticks(rotation=45, ha='right')
        plt.title('States with Highest and Lowest Climate Risk Scores')
        plt.ylabel('Risk Score (higher = more at risk)')
        plt.xlabel('')
        
        # Add rank labels on bars
        for i, p in enumerate(bars.patches):
            bars.annotate(f"#{combined.iloc[i]['rank']}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'bottom', 
                         xytext = (0, 5), textcoords = 'offset points')
        
        # 2. Distribution of risk scores
        plt.subplot(2, 1, 2)
        sns.histplot(self.risk_scores['risk_score'], bins=15, kde=True)
        plt.axvline(x=self.risk_scores['risk_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.risk_scores["risk_score"].mean():.1f}')
        plt.legend()
        plt.title('Distribution of Climate Risk Scores Across States')
        plt.xlabel('Risk Score')
        plt.ylabel('Count of States')
        
        plt.tight_layout()
        plt.savefig('climate_risk_rankings.png')
        plt.close()
        
        # 3. Create feature importance plot
        plt.figure(figsize=(12, 8))
        feature_importances = pd.Series(
            self.model.feature_importances_,
            index=self.features.columns
        ).sort_values(ascending=False).head(15)
        
        sns.barplot(x=feature_importances.values, y=feature_importances.index)
        plt.title('Top 15 Features for Climate Risk Prediction')
        plt.xlabel('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # 4. Create PCA plot to show state clustering
        plt.figure(figsize=(12, 10))
        
        # Perform PCA on the features
        pca = PCA(n_components=2)
        X_scaled = self.scaler.transform(self.features)
        pca_result = pca.fit_transform(X_scaled)
        
        # Create a dataframe with PCA results and risk categories
        pca_df = pd.DataFrame({
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1],
            'state': self.processed_data.index,
            'risk_score': self.risk_scores['risk_score'].values,
            'risk_category': self.risk_scores['risk_category'].values
        })
        
        # Plot states in PCA space, colored by risk category
        sns.scatterplot(x='PCA1', y='PCA2', hue='risk_category', 
                       size='risk_score', sizes=(50, 250),
                       palette='YlOrRd', data=pca_df)
        
        # Add state labels
        for _, row in pca_df.iterrows():
            plt.text(row['PCA1'] + 0.1, row['PCA2'] + 0.1, row['state'], 
                    fontsize=8, ha='left', va='bottom')
        
        plt.title('States Clustered by Climate Risk Factors (PCA)')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.tight_layout()
        plt.savefig('state_clustering.png')
        plt.close()
        
        print("Visualizations created and saved as image files.")
        
    def run_full_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("Running complete climate risk analysis pipeline...")
        self.fetch_data()
        self.preprocess_data()
        self.engineer_features()
        self.train_model()
        self.generate_risk_scores()
        self.visualize_results()
        
        # Print top 10 most at-risk states
        print("\n--- TOP 10 MOST AT-RISK STATES ---")
        top_risk_states = self.risk_scores.nsmallest(10, 'rank')
        for _, row in top_risk_states.iterrows():
            print(f"{row['rank']}. {row['state']}: {row['risk_score']:.1f}/100 ({row['risk_category']})")
        
        # Print 10 least at-risk states
        print("\n--- 10 LEAST AT-RISK STATES ---")
        bottom_risk_states = self.risk_scores.nlargest(10, 'rank')
        for _, row in bottom_risk_states.iterrows():
            print(f"{row['rank']}. {row['state']}: {row['risk_score']:.1f}/100 ({row['risk_category']})")
        
        print("\nAnalysis complete. For detailed insights on specific states, use the analyze_risk_factors('State Name') method.")
        
        return self.risk_scores

# Example usage
if __name__ == "__main__":
    model = ClimateRiskModel()
    scores = model.run_full_analysis()
    
    # Analyze a few specific states
    high_risk_state = scores.nsmallest(1, 'rank')['state'].values[0]
    low_risk_state = scores.nlargest(1, 'rank')['state'].values[0]
    
    model.analyze_risk_factors(high_risk_state)
    model.analyze_risk_factors(low_risk_state)
    model.analyze_risk_factors('California')