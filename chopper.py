# @title Community Health Outbreak Prevention System (CHOPS) - Final Working Version

# Install required packages


# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

class CHOPS:
    def __init__(self):
        self.data = None
        self.forecast = None
        self.risk_assessment = None
        self.model_performance = {}
    
    def load_data(self, upload=False):
        """Load data from file upload or use synthetic data"""
        if upload:
            uploaded = files.upload()
            file_name = next(iter(uploaded.keys()))
            self.data = pd.read_csv(io.StringIO(uploaded[file_name].decode('utf-8')))
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Loaded {len(self.data)} records from {file_name}")
        else:
            self._generate_synthetic_data()
        return self
    
    def _generate_synthetic_data(self):
        """Generate realistic synthetic health data"""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2023-01-01')
        regions = ['North', 'South', 'East', 'West']
        
        # Base cases with seasonality
        base = np.random.poisson(lam=5, size=len(dates))
        seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + 2 * np.cos(2 * np.pi * np.arange(len(dates)) / 7)

        # In _generate_synthetic_data():
        outbreaks = np.zeros(len(dates))
        # More concentrated outbreaks (3-5 days instead of 30)
        outbreaks[120:125] = np.random.poisson(lam=25, size=5)  # Spring outbreak
        outbreaks[240:245] = np.random.poisson(lam=20, size=5)  # Fall outbreak
        
        cases = np.tile(base + seasonal + outbreaks, len(regions)) + np.random.poisson(2, len(dates)*len(regions))
        
        self.data = pd.DataFrame({
            'date': np.repeat(dates, len(regions)),
            'region': np.tile(regions, len(dates)),
            'cases': cases
        })
        print("Generated synthetic data with regional variation")
        return self
    
    def preprocess_data(self):
        """Clean and engineer features"""
        # Remove any NA values that might cause issues
        self.data = self.data.dropna()
        
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['month'] = self.data['date'].dt.month
        
        # Normalize by region
        self.data['cases_norm'] = self.data.groupby('region')['cases'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # Rolling averages
        self.data['cases_7day'] = self.data.groupby('region')['cases'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        return self
    
    def detect_anomalies(self, method='isolation_forest'):
        """Detect anomalies using robust method"""
        # Ensure we only use complete cases
        clean_data = self.data.dropna(subset=['cases', 'cases_7day', 'cases_norm', 'day_of_week'])
        
        X = clean_data[['cases', 'cases_7day', 'cases_norm', 'day_of_week']].values
        
        if method == 'isolation_forest':
            clf = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = clf.fit_predict(X)
            
            # Merge results back to original data
            clean_data['anomaly_score'] = anomaly_scores
            clean_data['is_anomaly'] = (anomaly_scores == -1).astype(int)
            
            # Update main dataframe
            self.data = self.data.merge(
                clean_data[['date', 'region', 'anomaly_score', 'is_anomaly']],
                on=['date', 'region'],
                how='left'
            )
        
        self.model_performance['anomaly_detection'] = {
            'method': method,
            'anomalies_found': self.data['is_anomaly'].sum()
        }
        return self
    
    def forecast_cases(self):
        """Time series forecasting with Prophet"""
        results = []
        for region in self.data['region'].unique():
            region_data = self.data[self.data['region'] == region].dropna()
            prophet_df = region_data[['date', 'cases']].rename(columns={'date': 'ds', 'cases': 'y'})
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            forecast['region'] = region
            
            results.append(forecast)
        
        self.forecast = pd.concat(results)
        return self
    

    # In the assess_risk() method, replace the conditions with:

    def assess_risk(self):
        """Calculate outbreak risk levels with more realistic thresholds"""
        merged = pd.merge(
            self.data,
            self.forecast[['ds', 'region', 'yhat', 'yhat_lower', 'yhat_upper']],
            left_on=['date', 'region'],
            right_on=['ds', 'region'],
            how='left'
        ).dropna(subset=['yhat'])

        merged['deviation'] = merged['cases'] - merged['yhat']
        merged['deviation_pct'] = merged['deviation'] / merged['yhat']
    
        # More realistic risk classification
        conditions = [
            # High risk requires both large deviation AND anomaly
            (merged['deviation_pct'] > 1.0) & (merged['is_anomaly'] == 1),
        
            # Medium risk requires either large deviation OR anomaly
            ((merged['deviation_pct'] > 0.7) & (merged['is_anomaly'] == 1)) |
            ((merged['deviation_pct'] > 1.0) & (merged['is_anomaly'] == 0)),
        
            # Low risk - moderate deviation
            (merged['deviation_pct'] > 0.5)
        ]
        choices = ['High', 'Medium', 'Low']
        merged['risk_level'] = np.select(conditions, choices, default='Normal')
    
        # Only keep consecutive high-risk days
        merged['risk_group'] = (merged['risk_level'] != 'High').cumsum()
        risk_counts = merged[merged['risk_level'] == 'High'].groupby('risk_group').size()
    
        # Filter out single-day "high risk" alerts
        single_day_groups = risk_counts[risk_counts < 2].index
        merged.loc[merged['risk_group'].isin(single_day_groups), 'risk_level'] = 'Medium'
    
        self.risk_assessment = merged
        return self    
    
    def visualize_results(self):
        """Create interactive dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Case Trends with Anomalies", "Risk Level Distribution")
        )
        
        # Plot 1: Case trends with anomalies
        for region in self.data['region'].unique():
            region_data = self.data[self.data['region'] == region]
            fig.add_trace(
                go.Scatter(
                    x=region_data['date'],
                    y=region_data['cases'],
                    name=f'{region} Cases',
                    mode='lines'
                ),
                row=1, col=1
            )
            
            anomalies = region_data[region_data['is_anomaly'] == 1]
            fig.add_trace(
                go.Scatter(
                    x=anomalies['date'],
                    y=anomalies['cases'],
                    mode='markers',
                    name=f'{region} Anomaly',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
        
        # Plot 2: Risk level distribution
        risk_counts = self.risk_assessment['risk_level'].value_counts()
        fig.add_trace(
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=['red', 'orange', 'yellow', 'green'],
                name='Risk Levels'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Community Health Outbreak Prevention Dashboard"
        )
        return fig
    
    def generate_report(self):
        """Generate summary statistics and recommendations"""
        report = {
            "time_period": {
                "start": self.data['date'].min().strftime('%Y-%m-%d'),
                "end": self.data['date'].max().strftime('%Y-%m-%d')
            },
            "total_cases": int(self.data['cases'].sum()),
            "anomalies_detected": int(self.data['is_anomaly'].sum()),
            "high_risk_periods": len(self.risk_assessment[self.risk_assessment['risk_level'] == 'High']),
            "recommendations": self._generate_recommendations()
        }
        return report
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        high_risk = self.risk_assessment[self.risk_assessment['risk_level'] == 'High']
        recommendations = []
        
        if not high_risk.empty:
            for _, group in high_risk.groupby(['date', 'region']):
                rec = {
                    "date": group['date'].iloc[0].strftime('%Y-%m-%d'),
                    "region": group['region'].iloc[0],
                    "case_count": int(group['cases'].mean()),
                    "expected_cases": round(group['yhat'].mean(), 1),
                    "deviation": f"{round(group['deviation_pct'].mean() * 100, 1)}%",
                    "actions": [
                        "Increase testing and surveillance",
                        "Alert nearby healthcare facilities",
                        "Review inventory of essential medicines"
                    ]
                }
                recommendations.append(rec)
        
        return recommendations if recommendations else ["No high-risk periods detected"]

# ======================
# Execute the full system
# ======================
print("Initializing Community Health Outbreak Prevention System...\n")

# Initialize and run pipeline
chops = CHOPS()
(
    chops.load_data(upload=False)  # Set upload=True to use your own data
    .preprocess_data()
    .detect_anomalies(method='isolation_forest')
    .forecast_cases()
    .assess_risk()
)

# Display results
print("\n=== System Report ===")
report = chops.generate_report()
for key, value in report.items():
    if key != 'recommendations':
        print(f"{key.replace('_', ' ').title()}: {value}")

print("\n=== Recommendations ===")
for rec in report['recommendations']:
    if isinstance(rec, dict):
        print(f"\n🚨 {rec['date']} - {rec['region']}:")
        print(f"   Cases: {rec['case_count']} (Expected: {rec['expected_cases']}, Deviation: {rec['deviation']})")
        for action in rec['actions']:
            print(f"   • {action}")
    else:
        print(rec)

# Show interactive dashboard
print("\nGenerating interactive dashboard...")
fig = chops.visualize_results()
fig.show()