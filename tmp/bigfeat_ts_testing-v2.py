"""
Comprehensive Testing Script for BigFeat Time Series Operations

This script tests BigFeat's time series feature engineering capabilities across
multiple datasets with various characteristics, comparing different modes and
generating comprehensive performance visualizations.

Author: BigFeat Testing Suite
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class BigFeatTimeSeriesTester:
    """Comprehensive tester for BigFeat time series operations"""

    def __init__(self, output_dir='bigfeat_test_results'):
        """
        Initialize the tester

        Parameters:
        -----------
        output_dir : str
            Directory to save results and plots
        """
        self.output_dir = output_dir
        self.results = []
        self.detailed_metrics = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print("BigFeat Time Series Operations - Comprehensive Testing Suite")
        print("=" * 80)

    def generate_synthetic_datasets(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Generate multiple synthetic datasets with different characteristics

        Returns:
        --------
        dict
            Dictionary of dataset_name -> (X, y) pairs
        """
        datasets = {}

        print("\n📊 Generating Synthetic Datasets...")

        # 1. Strong Daily Periodicity (e.g., website traffic)
        print("  ✓ Daily periodic pattern (7-day cycle)")
        n_samples = 500
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        # Create strong weekly pattern
        daily_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 7)
        trend = np.linspace(0, 5, n_samples)
        noise = np.random.normal(0, 0.2, n_samples)

        feature1 = daily_pattern + trend + noise
        feature2 = 0.5 * daily_pattern + 0.3 * trend + np.random.normal(0, 0.15, n_samples)
        feature3 = np.random.normal(0, 1, n_samples)  # Random feature for contrast

        X_daily = pd.DataFrame({
            'datetime': dates,
            'traffic': feature1,
            'engagement': feature2,
            'random_noise': feature3
        })

        # Binary target based on threshold
        y_daily = (feature1 + feature2 > 1).astype(int)
        datasets['daily_periodic'] = (X_daily, y_daily)

        # 2. Monthly Periodicity (e.g., sales data)
        print("  ✓ Monthly periodic pattern (30-day cycle)")
        n_samples = 600
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        monthly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 30)
        trend = np.linspace(0, 3, n_samples)
        noise = np.random.normal(0, 0.3, n_samples)

        feature1 = monthly_pattern * 2 + trend + noise
        feature2 = 0.7 * monthly_pattern + 0.2 * trend + np.random.normal(0, 0.2, n_samples)

        X_monthly = pd.DataFrame({
            'datetime': dates,
            'sales': feature1,
            'marketing_spend': feature2,
            'seasonality_proxy': np.cos(2 * np.pi * np.arange(n_samples) / 365)
        })

        y_monthly = (feature1 + feature2 > 1.5).astype(int)
        datasets['monthly_periodic'] = (X_monthly, y_monthly)

        # 3. Multi-scale Periodicity (daily + weekly + monthly)
        print("  ✓ Multi-scale periodic pattern (7 + 30 day cycles)")
        n_samples = 700
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        daily = 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        weekly = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 14)
        monthly = 0.7 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        trend = np.linspace(0, 4, n_samples)
        noise = np.random.normal(0, 0.25, n_samples)

        feature1 = daily + weekly + monthly + trend + noise
        feature2 = 0.5 * daily + 0.3 * monthly + np.random.normal(0, 0.2, n_samples)

        X_multiscale = pd.DataFrame({
            'datetime': dates,
            'value': feature1,
            'auxiliary': feature2,
            'trend_proxy': trend
        })

        y_multiscale = (feature1 + feature2 > 2).astype(int)
        datasets['multiscale_periodic'] = (X_multiscale, y_multiscale)

        # 4. Weak Periodicity (noisy data)
        print("  ✓ Weak periodic pattern (high noise)")
        n_samples = 500
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        weak_pattern = 0.2 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        strong_noise = np.random.normal(0, 1, n_samples)
        trend = np.linspace(0, 2, n_samples)

        feature1 = weak_pattern + strong_noise + trend
        feature2 = 0.1 * weak_pattern + np.random.normal(0, 0.8, n_samples)

        X_weak = pd.DataFrame({
            'datetime': dates,
            'noisy_signal': feature1,
            'very_noisy': feature2,
            'pure_random': np.random.normal(0, 1, n_samples)
        })

        y_weak = (feature1 + feature2 > 1).astype(int)
        datasets['weak_periodic'] = (X_weak, y_weak)

        # 5. Non-periodic Trend (pure trend, no periodicity)
        print("  ✓ Non-periodic trend (control)")
        n_samples = 500
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        trend = np.linspace(0, 5, n_samples)
        noise = np.random.normal(0, 0.5, n_samples)

        feature1 = trend + noise
        feature2 = 0.7 * trend + np.random.normal(0, 0.3, n_samples)

        X_nonperiodic = pd.DataFrame({
            'datetime': dates,
            'trend_feature': feature1,
            'correlated_trend': feature2,
            'random': np.random.normal(0, 1, n_samples)
        })

        y_nonperiodic = (feature1 + feature2 > 3).astype(int)
        datasets['non_periodic'] = (X_nonperiodic, y_nonperiodic)

        # 6. Regression Dataset with Strong Seasonality
        print("  ✓ Regression with seasonal pattern")
        n_samples = 600
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        trend = np.linspace(10, 50, n_samples)
        noise = np.random.normal(0, 2, n_samples)

        feature1 = seasonal + trend + noise
        feature2 = 0.5 * seasonal + 0.3 * trend + np.random.normal(0, 1.5, n_samples)

        X_regression = pd.DataFrame({
            'datetime': dates,
            'temperature': feature1,
            'humidity': feature2,
            'pressure': np.random.normal(1000, 20, n_samples)
        })

        # Continuous target
        y_regression = feature1 * 1.5 + feature2 * 0.8 + np.random.normal(0, 3, n_samples)
        datasets['regression_seasonal'] = (X_regression, y_regression)

        print(f"\n✓ Generated {len(datasets)} datasets")
        return datasets

    def load_real_datasets(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Placeholder for loading real-world datasets

        Returns:
        --------
        dict
            Dictionary of real dataset_name -> (X, y) pairs
        """
        # In practice, you would load real datasets here
        # For now, return empty dict - users can add their own
        return {}

    def test_configuration(self,
                           X: pd.DataFrame,
                           y: np.ndarray,
                           dataset_name: str,
                           task_type: str,
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a specific BigFeat configuration

        Parameters:
        -----------
        X : DataFrame
            Input features with datetime column
        y : ndarray
            Target variable
        dataset_name : str
            Name of the dataset
        task_type : str
            'classification' or 'regression'
        config : dict
            BigFeat configuration parameters

        Returns:
        --------
        dict
            Results dictionary with metrics and timing
        """
        from bigfeat.bigfeat_base import BigFeat

        config_name = config['name']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Initialize BigFeat
        start_time = time.time()

        bf = BigFeat(
            task_type=task_type,
            enable_time_series=config.get('enable_time_series', 'no'),
            datetime_col=config.get('datetime_col', None),
            window_sizes=config.get('window_sizes', None),
            lag_periods=config.get('lag_periods', None),
            verbose=False,
            dft_confidence_threshold=config.get('dft_confidence_threshold', 0.3),
            dft_min_window_days=config.get('dft_min_window_days', 3),
            dft_max_window_days=config.get('dft_max_window_days', 365),
            dft_n_windows=config.get('dft_n_windows', 6)
        )

        # Fit and transform
        try:
            X_train_transformed = bf.fit(
                X_train, y_train,
                gen_size=5,
                iterations=3,
                random_state=42
            )

            X_test_transformed = bf.transform(X_test)

            fit_time = time.time() - start_time

            # Train model
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_transformed, y_train)

                y_pred = model.predict(X_test_transformed)
                y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = None

                score = accuracy
                score_name = 'accuracy'

            else:  # regression
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_transformed, y_train)

                y_pred = model.predict(X_test_transformed)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                score = r2
                score_name = 'r2'
                auc = None

            # Count time series operations used
            ts_ops_count = 0
            if bf.enable_time_series and hasattr(bf, 'tracking_ops'):
                for ops in bf.tracking_ops:
                    for op_info in ops:
                        if len(op_info) > 0 and callable(op_info[0]):
                            if hasattr(bf, 'time_series_operators') and op_info[0] in bf.time_series_operators:
                                ts_ops_count += 1

            # Get DFT info
            dft_info = bf.get_dft_summary() if hasattr(bf, 'get_dft_summary') else {}

            result = {
                'dataset': dataset_name,
                'config': config_name,
                'task_type': task_type,
                'score': score,
                'score_name': score_name,
                'auc': auc,
                'fit_time': fit_time,
                'n_features': X_train_transformed.shape[1],
                'n_original_features': X_train.shape[1] - 1,  # Exclude datetime
                'ts_enabled': bf.enable_time_series,
                'ts_ops_count': ts_ops_count,
                'dft_detection_strategy': dft_info.get('detection_strategy', None),
                'dft_avg_confidence': dft_info.get('avg_confidence', None),
                'window_sizes': dft_info.get('window_sizes', None),
                'success': True,
                'error': None
            }

            if task_type == 'regression':
                result['rmse'] = rmse

        except Exception as e:
            result = {
                'dataset': dataset_name,
                'config': config_name,
                'task_type': task_type,
                'success': False,
                'error': str(e)
            }

        return result

    def run_comprehensive_tests(self):
        """Run comprehensive tests across all datasets and configurations"""

        # Generate datasets
        datasets = self.generate_synthetic_datasets()
        real_datasets = self.load_real_datasets()
        datasets.update(real_datasets)

        # Define test configurations
        configs = [
            {
                'name': 'TS_Disabled',
                'enable_time_series': 'no',
                'datetime_col': None
            },
            {
                'name': 'TS_Auto',
                'enable_time_series': 'auto',
                'datetime_col': 'datetime',
                'dft_confidence_threshold': 0.3
            },
            {
                'name': 'TS_Forced',
                'enable_time_series': 'yes',
                'datetime_col': 'datetime'
            },
            {
                'name': 'TS_Auto_Strict',
                'enable_time_series': 'auto',
                'datetime_col': 'datetime',
                'dft_confidence_threshold': 0.5
            },
            {
                'name': 'TS_Manual_Windows',
                'enable_time_series': 'yes',
                'datetime_col': 'datetime',
                'window_sizes': ['7D', '14D', '30D'],
                'lag_periods': ['1D', '7D']
            }
        ]

        print("\n" + "=" * 80)
        print("Running Comprehensive Tests")
        print("=" * 80)

        total_tests = len(datasets) * len(configs)
        current_test = 0

        for dataset_name, (X, y) in datasets.items():
            # Determine task type
            if dataset_name == 'regression_seasonal':
                task_type = 'regression'
            else:
                task_type = 'classification'

            print(f"\n📊 Testing Dataset: {dataset_name} ({task_type})")
            print(
                f"   Shape: {X.shape}, Target distribution: {np.bincount(y.astype(int)) if task_type == 'classification' else f'range=[{y.min():.2f}, {y.max():.2f}]'}")

            for config in configs:
                current_test += 1
                print(f"\n  [{current_test}/{total_tests}] Config: {config['name']}", end=" ... ")

                result = self.test_configuration(X, y, dataset_name, task_type, config)
                self.results.append(result)

                if result['success']:
                    score_str = f"{result['score']:.4f}"
                    time_str = f"{result['fit_time']:.2f}s"
                    ts_str = f"TS:{result['ts_ops_count']}" if result['ts_enabled'] else "No TS"
                    print(f"✓ {result['score_name']}={score_str}, time={time_str}, {ts_str}")
                else:
                    print(f"✗ Error: {result['error']}")

        print("\n" + "=" * 80)
        print("All Tests Completed!")
        print("=" * 80)

    def generate_visualizations(self):
        """Generate comprehensive visualizations of test results"""

        df = pd.DataFrame(self.results)
        df_success = df[df['success'] == True].copy()

        if len(df_success) == 0:
            print("\n⚠️  No successful tests to visualize")
            return

        print("\n📈 Generating Visualizations...")

        # 1. Performance Comparison by Configuration
        self._plot_performance_comparison(df_success)

        # 2. Time Series vs Non-Time Series Performance
        self._plot_ts_vs_nots_performance(df_success)

        # 3. Configuration Performance by Dataset
        self._plot_config_by_dataset(df_success)

        # 4. Fit Time Analysis
        self._plot_fit_time_analysis(df_success)

        # 5. Feature Count Analysis
        self._plot_feature_count_analysis(df_success)

        # 6. DFT Detection Analysis
        self._plot_dft_detection_analysis(df_success)

        # 7. Performance Heatmap
        self._plot_performance_heatmap(df_success)

        # 8. Time Series Operations Usage
        self._plot_ts_operations_usage(df_success)

        print(f"✓ All plots saved to {self.output_dir}/")

    def _plot_performance_comparison(self, df):
        """Plot 1: Overall performance comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Classification datasets
        df_class = df[df['task_type'] == 'classification']
        if len(df_class) > 0:
            sns.boxplot(data=df_class, x='config', y='score', ax=axes[0])
            axes[0].set_title('Classification Performance by Configuration', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Configuration', fontsize=12)
            axes[0].set_ylabel('Accuracy', fontsize=12)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)

        # Regression datasets
        df_reg = df[df['task_type'] == 'regression']
        if len(df_reg) > 0:
            sns.boxplot(data=df_reg, x='config', y='score', ax=axes[1])
            axes[1].set_title('Regression Performance by Configuration', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Configuration', fontsize=12)
            axes[1].set_ylabel('R² Score', fontsize=12)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Performance comparison plot saved")

    def _plot_ts_vs_nots_performance(self, df):
        """Plot 2: Time series vs non-time series performance"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Add TS enabled column
        df['ts_category'] = df['ts_enabled'].map({True: 'TS Enabled', False: 'TS Disabled'})

        # By dataset - classification
        df_class = df[df['task_type'] == 'classification']
        if len(df_class) > 0:
            pivot_class = df_class.pivot_table(
                values='score',
                index='dataset',
                columns='ts_category',
                aggfunc='mean'
            )
            pivot_class.plot(kind='bar', ax=axes[0, 0], rot=45)
            axes[0, 0].set_title('Classification: TS vs Non-TS by Dataset', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Mean Accuracy', fontsize=11)
            axes[0, 0].legend(title='')
            axes[0, 0].grid(axis='y', alpha=0.3)

        # By dataset - regression
        df_reg = df[df['task_type'] == 'regression']
        if len(df_reg) > 0:
            pivot_reg = df_reg.pivot_table(
                values='score',
                index='dataset',
                columns='ts_category',
                aggfunc='mean'
            )
            pivot_reg.plot(kind='bar', ax=axes[0, 1], rot=45)
            axes[0, 1].set_title('Regression: TS vs Non-TS by Dataset', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Mean R² Score', fontsize=11)
            axes[0, 1].legend(title='')
            axes[0, 1].grid(axis='y', alpha=0.3)

        # Overall comparison
        overall_comparison = df.groupby('ts_category')['score'].agg(['mean', 'std'])
        axes[1, 0].bar(overall_comparison.index, overall_comparison['mean'],
                       yerr=overall_comparison['std'], capsize=10, alpha=0.7)
        axes[1, 0].set_title('Overall Performance: TS vs Non-TS', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Mean Score', fontsize=11)
        axes[1, 0].grid(axis='y', alpha=0.3)

        # TS operations count vs performance
        df_ts = df[df['ts_enabled'] == True]
        if len(df_ts) > 0:
            axes[1, 1].scatter(df_ts['ts_ops_count'], df_ts['score'], alpha=0.6, s=100)
            axes[1, 1].set_xlabel('Number of TS Operations Used', fontsize=11)
            axes[1, 1].set_ylabel('Score', fontsize=11)
            axes[1, 1].set_title('TS Operations Usage vs Performance', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)

            # Add trend line
            z = np.polyfit(df_ts['ts_ops_count'], df_ts['score'], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(df_ts['ts_ops_count'], p(df_ts['ts_ops_count']),
                            "r--", alpha=0.8, label='Trend')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_ts_vs_nots_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ TS vs Non-TS comparison plot saved")

    def _plot_config_by_dataset(self, df):
        """Plot 3: Configuration performance by dataset"""
        datasets = df['dataset'].unique()
        n_datasets = len(datasets)
        n_cols = 3
        n_rows = (n_datasets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten() if n_datasets > 1 else [axes]

        for idx, dataset in enumerate(datasets):
            df_dataset = df[df['dataset'] == dataset]

            # Plot bars
            configs = df_dataset['config'].unique()
            scores = [df_dataset[df_dataset['config'] == c]['score'].mean() for c in configs]
            colors = ['green' if df_dataset[df_dataset['config'] == c]['ts_enabled'].any()
                      else 'blue' for c in configs]

            axes[idx].bar(range(len(configs)), scores, color=colors, alpha=0.7)
            axes[idx].set_xticks(range(len(configs)))
            axes[idx].set_xticklabels(configs, rotation=45, ha='right')
            axes[idx].set_title(f'Dataset: {dataset}', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].grid(axis='y', alpha=0.3)

            # Add value labels
            for i, v in enumerate(scores):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_config_by_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Configuration by dataset plot saved")

    def _plot_fit_time_analysis(self, df):
        """Plot 4: Fit time analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Time by configuration
        time_by_config = df.groupby('config')['fit_time'].agg(['mean', 'std'])
        axes[0, 0].bar(time_by_config.index, time_by_config['mean'],
                       yerr=time_by_config['std'], capsize=5, alpha=0.7)
        axes[0, 0].set_title('Mean Fit Time by Configuration', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Time (seconds)', fontsize=11)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Time by dataset
        time_by_dataset = df.groupby('dataset')['fit_time'].mean().sort_values()
        axes[0, 1].barh(range(len(time_by_dataset)), time_by_dataset.values, alpha=0.7)
        axes[0, 1].set_yticks(range(len(time_by_dataset)))
        axes[0, 1].set_yticklabels(time_by_dataset.index)
        axes[0, 1].set_title('Mean Fit Time by Dataset', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time (seconds)', fontsize=11)
        axes[0, 1].grid(axis='x', alpha=0.3)

        # TS vs Non-TS time
        time_by_ts = df.groupby('ts_enabled')['fit_time'].agg(['mean', 'std'])
        x_labels = ['TS Disabled', 'TS Enabled']
        axes[1, 0].bar(x_labels, time_by_ts['mean'], yerr=time_by_ts['std'],
                       capsize=10, alpha=0.7, color=['blue', 'green'])
        axes[1, 0].set_title('Fit Time: TS vs Non-TS', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Time (seconds)', fontsize=11)
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Time vs Performance scatter
        axes[1, 1].scatter(df['fit_time'], df['score'],
                           c=df['ts_enabled'].map({True: 'green', False: 'blue'}),
                           alpha=0.6, s=100)
        axes[1, 1].set_xlabel('Fit Time (seconds)', fontsize=11)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Fit Time vs Performance', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='TS Enabled'),
                           Patch(facecolor='blue', label='TS Disabled')]
        axes[1, 1].legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_fit_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Fit time analysis plot saved")

    def _plot_feature_count_analysis(self, df):
        """Plot 5: Feature count analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Features generated by configuration
        feature_stats = df.groupby('config')['n_features'].agg(['mean', 'std'])
        axes[0, 0].bar(feature_stats.index, feature_stats['mean'],
                       yerr=feature_stats['std'], capsize=5, alpha=0.7)
        axes[0, 0].set_title('Features Generated by Configuration', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Features', fontsize=11)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # TS vs Non-TS feature count
        ts_features = df.groupby('ts_enabled')['n_features'].mean()
        axes[0, 1].bar(['TS Disabled', 'TS Enabled'], ts_features.values,
                       alpha=0.7, color=['blue', 'green'])
        axes[0, 1].set_title('Feature Count: TS vs Non-TS', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Mean Number of Features', fontsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Feature count vs performance
        axes[1, 0].scatter(df['n_features'], df['score'],
                           c=df['ts_enabled'].map({True: 'green', False: 'blue'}),
                           alpha=0.6, s=100)
        axes[1, 0].set_xlabel('Number of Features', fontsize=11)
        axes[1, 0].set_ylabel('Score', fontsize=11)
        axes[1, 0].set_title('Feature Count vs Performance', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='TS Enabled'),
                           Patch(facecolor='blue', label='TS Disabled')]
        axes[1, 0].legend(handles=legend_elements)

        # Feature expansion ratio
        df['expansion_ratio'] = df['n_features'] / df['n_original_features']
        expansion_by_config = df.groupby('config')['expansion_ratio'].mean().sort_values()
        axes[1, 1].barh(range(len(expansion_by_config)), expansion_by_config.values, alpha=0.7)
        axes[1, 1].set_yticks(range(len(expansion_by_config)))
        axes[1, 1].set_yticklabels(expansion_by_config.index)
        axes[1, 1].set_title('Feature Expansion Ratio by Configuration', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Expansion Ratio (Final/Original)', fontsize=11)
        axes[1, 1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_feature_count_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Feature count analysis plot saved")

    def _plot_dft_detection_analysis(self, df):
        """Plot 6: DFT detection analysis"""
        # Filter for rows with DFT info
        df_dft = df[df['dft_avg_confidence'].notna()].copy()

        if len(df_dft) == 0:
            print("  ⚠️  No DFT detection data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Confidence scores by dataset
        confidence_by_dataset = df_dft.groupby('dataset')['dft_avg_confidence'].mean().sort_values()
        axes[0, 0].barh(range(len(confidence_by_dataset)), confidence_by_dataset.values, alpha=0.7)
        axes[0, 0].set_yticks(range(len(confidence_by_dataset)))
        axes[0, 0].set_yticklabels(confidence_by_dataset.index)
        axes[0, 0].set_title('DFT Confidence Score by Dataset', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Mean Confidence Score', fontsize=11)
        axes[0, 0].axvline(x=0.3, color='red', linestyle='--', label='Default Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)

        # Confidence vs Performance
        axes[0, 1].scatter(df_dft['dft_avg_confidence'], df_dft['score'],
                           alpha=0.6, s=100, c='purple')
        axes[0, 1].set_xlabel('DFT Confidence Score', fontsize=11)
        axes[0, 1].set_ylabel('Model Score', fontsize=11)
        axes[0, 1].set_title('DFT Confidence vs Model Performance', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # Add trend line
        if len(df_dft) > 1:
            z = np.polyfit(df_dft['dft_avg_confidence'], df_dft['score'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_dft['dft_avg_confidence'].min(),
                                 df_dft['dft_avg_confidence'].max(), 100)
            axes[0, 1].plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
            axes[0, 1].legend()

        # Detection strategy distribution
        strategy_counts = df_dft['dft_detection_strategy'].value_counts()
        axes[1, 0].pie(strategy_counts.values, labels=strategy_counts.index,
                       autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('DFT Detection Strategy Distribution', fontsize=12, fontweight='bold')

        # Performance by detection strategy
        if len(df_dft['dft_detection_strategy'].unique()) > 1:
            sns.boxplot(data=df_dft, x='dft_detection_strategy', y='score', ax=axes[1, 1])
            axes[1, 1].set_title('Performance by DFT Strategy', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Detection Strategy', fontsize=11)
            axes[1, 1].set_ylabel('Score', fontsize=11)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor strategy comparison',
                            ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_dft_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ DFT detection analysis plot saved")

    def _plot_performance_heatmap(self, df):
        """Plot 7: Performance heatmap"""
        # Create pivot table
        pivot = df.pivot_table(
            values='score',
            index='dataset',
            columns='config',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=pivot.values.mean(), ax=ax, cbar_kws={'label': 'Score'})

        ax.set_title('Performance Heatmap: Dataset vs Configuration',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Performance heatmap saved")

    def _plot_ts_operations_usage(self, df):
        """Plot 8: Time series operations usage"""
        df_ts = df[df['ts_enabled'] == True].copy()

        if len(df_ts) == 0:
            print("  ⚠️  No time series operations data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # TS operations by config
        ts_ops_by_config = df_ts.groupby('config')['ts_ops_count'].mean().sort_values()
        axes[0, 0].barh(range(len(ts_ops_by_config)), ts_ops_by_config.values, alpha=0.7)
        axes[0, 0].set_yticks(range(len(ts_ops_by_config)))
        axes[0, 0].set_yticklabels(ts_ops_by_config.index)
        axes[0, 0].set_title('Mean TS Operations by Configuration', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Number of TS Operations', fontsize=11)
        axes[0, 0].grid(axis='x', alpha=0.3)

        # TS operations by dataset
        ts_ops_by_dataset = df_ts.groupby('dataset')['ts_ops_count'].mean().sort_values()
        axes[0, 1].barh(range(len(ts_ops_by_dataset)), ts_ops_by_dataset.values, alpha=0.7, color='green')
        axes[0, 1].set_yticks(range(len(ts_ops_by_dataset)))
        axes[0, 1].set_yticklabels(ts_ops_by_dataset.index)
        axes[0, 1].set_title('Mean TS Operations by Dataset', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Number of TS Operations', fontsize=11)
        axes[0, 1].grid(axis='x', alpha=0.3)

        # Distribution of TS operations
        axes[1, 0].hist(df_ts['ts_ops_count'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('Distribution of TS Operations Count', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Number of TS Operations', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].grid(axis='y', alpha=0.3)

        # TS operations vs score improvement
        # Calculate score improvement vs baseline (TS_Disabled)
        improvements = []
        for dataset in df_ts['dataset'].unique():
            df_dataset = df[df['dataset'] == dataset]
            baseline = df_dataset[df_dataset['config'] == 'TS_Disabled']['score'].mean()

            for _, row in df_dataset[df_dataset['ts_enabled']].iterrows():
                improvement = row['score'] - baseline
                improvements.append({
                    'ts_ops_count': row['ts_ops_count'],
                    'improvement': improvement
                })

        if improvements:
            imp_df = pd.DataFrame(improvements)
            axes[1, 1].scatter(imp_df['ts_ops_count'], imp_df['improvement'],
                               alpha=0.6, s=100, color='purple')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Number of TS Operations', fontsize=11)
            axes[1, 1].set_ylabel('Score Improvement vs Baseline', fontsize=11)
            axes[1, 1].set_title('TS Operations vs Performance Improvement', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_ts_operations_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ TS operations usage plot saved")

    def generate_summary_report(self):
        """Generate a comprehensive text summary report"""

        df = pd.DataFrame(self.results)
        df_success = df[df['success'] == True]

        report_path = f'{self.output_dir}/SUMMARY_REPORT.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BIGFEAT TIME SERIES OPERATIONS - COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests Run: {len(df)}\n")
            f.write(f"Successful Tests: {len(df_success)}\n")
            f.write(f"Failed Tests: {len(df[df['success'] == False])}\n\n")

            # Overall statistics
            f.write("-" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Mean Score (All Tests): {df_success['score'].mean():.4f} ± {df_success['score'].std():.4f}\n")
            f.write(f"Mean Fit Time: {df_success['fit_time'].mean():.2f}s ± {df_success['fit_time'].std():.2f}s\n")
            f.write(
                f"Mean Features Generated: {df_success['n_features'].mean():.1f} ± {df_success['n_features'].std():.1f}\n\n")

            # Time Series vs Non-Time Series
            f.write("-" * 80 + "\n")
            f.write("TIME SERIES vs NON-TIME SERIES COMPARISON\n")
            f.write("-" * 80 + "\n\n")

            ts_stats = df_success.groupby('ts_enabled')['score'].agg(['mean', 'std', 'count'])
            f.write("Performance:\n")
            f.write(
                f"  TS Disabled: {ts_stats.loc[False, 'mean']:.4f} ± {ts_stats.loc[False, 'std']:.4f} (n={ts_stats.loc[False, 'count']:.0f})\n")
            if True in ts_stats.index:
                f.write(
                    f"  TS Enabled:  {ts_stats.loc[True, 'mean']:.4f} ± {ts_stats.loc[True, 'std']:.4f} (n={ts_stats.loc[True, 'count']:.0f})\n")
                improvement = ((ts_stats.loc[True, 'mean'] - ts_stats.loc[False, 'mean']) / ts_stats.loc[
                    False, 'mean']) * 100
                f.write(f"  Improvement: {improvement:+.2f}%\n\n")

            time_stats = df_success.groupby('ts_enabled')['fit_time'].agg(['mean', 'std'])
            f.write("Fit Time:\n")
            f.write(f"  TS Disabled: {time_stats.loc[False, 'mean']:.2f}s ± {time_stats.loc[False, 'std']:.2f}s\n")
            if True in time_stats.index:
                f.write(f"  TS Enabled:  {time_stats.loc[True, 'mean']:.2f}s ± {time_stats.loc[True, 'std']:.2f}s\n")
                overhead = ((time_stats.loc[True, 'mean'] - time_stats.loc[False, 'mean']) / time_stats.loc[
                    False, 'mean']) * 100
                f.write(f"  Overhead: {overhead:+.2f}%\n\n")

            # Configuration performance
            f.write("-" * 80 + "\n")
            f.write("CONFIGURATION PERFORMANCE RANKING\n")
            f.write("-" * 80 + "\n\n")

            config_stats = df_success.groupby('config').agg({
                'score': ['mean', 'std'],
                'fit_time': 'mean',
                'ts_ops_count': 'mean'
            }).round(4)

            config_stats = config_stats.sort_values(('score', 'mean'), ascending=False)

            f.write(f"{'Rank':<6} {'Configuration':<20} {'Score':<15} {'Time(s)':<10} {'TS Ops':<8}\n")
            f.write("-" * 80 + "\n")

            for idx, (config, row) in enumerate(config_stats.iterrows(), 1):
                score_str = f"{row[('score', 'mean')]:.4f}±{row[('score', 'std')]:.4f}"
                time_str = f"{row[('fit_time', 'mean')]:.2f}"
                ts_ops = f"{row[('ts_ops_count', 'mean')]:.1f}" if not pd.isna(row[('ts_ops_count', 'mean')]) else "N/A"
                f.write(f"{idx:<6} {config:<20} {score_str:<15} {time_str:<10} {ts_ops:<8}\n")

            f.write("\n")

            # Dataset performance
            f.write("-" * 80 + "\n")
            f.write("DATASET PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n\n")

            for dataset in df_success['dataset'].unique():
                df_dataset = df_success[df_success['dataset'] == dataset]
                f.write(f"\n{dataset}:\n")
                f.write(f"  Best Config: {df_dataset.loc[df_dataset['score'].idxmax(), 'config']}\n")
                f.write(f"  Best Score: {df_dataset['score'].max():.4f}\n")
                f.write(f"  Mean Score: {df_dataset['score'].mean():.4f} ± {df_dataset['score'].std():.4f}\n")

                # TS improvement for this dataset
                baseline = df_dataset[df_dataset['config'] == 'TS_Disabled']['score'].values
                ts_scores = df_dataset[df_dataset['ts_enabled'] == True]['score'].values
                if len(baseline) > 0 and len(ts_scores) > 0:
                    improvement = ((ts_scores.mean() - baseline[0]) / baseline[0]) * 100
                    f.write(f"  TS Improvement: {improvement:+.2f}%\n")

            # DFT Analysis
            df_dft = df_success[df_success['dft_avg_confidence'].notna()]
            if len(df_dft) > 0:
                f.write("\n" + "-" * 80 + "\n")
                f.write("DFT DETECTION ANALYSIS\n")
                f.write("-" * 80 + "\n\n")

                f.write(f"Tests with DFT: {len(df_dft)}\n")
                f.write(
                    f"Mean Confidence: {df_dft['dft_avg_confidence'].mean():.4f} ± {df_dft['dft_avg_confidence'].std():.4f}\n")
                f.write(
                    f"Confidence Range: [{df_dft['dft_avg_confidence'].min():.4f}, {df_dft['dft_avg_confidence'].max():.4f}]\n\n")

                f.write("Detection Strategies:\n")
                strategy_counts = df_dft['dft_detection_strategy'].value_counts()
                for strategy, count in strategy_counts.items():
                    pct = (count / len(df_dft)) * 100
                    f.write(f"  {strategy}: {count} ({pct:.1f}%)\n")

            # Key insights
            f.write("\n" + "=" * 80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("=" * 80 + "\n\n")

            # Best configuration overall
            best_config = df_success.groupby('config')['score'].mean().idxmax()
            best_score = df_success.groupby('config')['score'].mean().max()
            f.write(f"1. Best Overall Configuration: {best_config} (score={best_score:.4f})\n")

            # TS effectiveness
            if True in ts_stats.index and False in ts_stats.index:
                ts_better = ts_stats.loc[True, 'mean'] > ts_stats.loc[False, 'mean']
                if ts_better:
                    f.write(f"2. Time series features IMPROVE performance by {improvement:.2f}%\n")
                else:
                    f.write(f"2. Time series features DEGRADE performance by {-improvement:.2f}%\n")

            # Most improved dataset
            improvements_by_dataset = {}
            for dataset in df_success['dataset'].unique():
                df_dataset = df_success[df_success['dataset'] == dataset]
                baseline = df_dataset[df_dataset['config'] == 'TS_Disabled']['score'].values
                ts_scores = df_dataset[df_dataset['ts_enabled'] == True]['score'].values
                if len(baseline) > 0 and len(ts_scores) > 0:
                    improvements_by_dataset[dataset] = ((ts_scores.mean() - baseline[0]) / baseline[0]) * 100

            if improvements_by_dataset:
                best_dataset = max(improvements_by_dataset, key=improvements_by_dataset.get)
                best_improvement = improvements_by_dataset[best_dataset]
                f.write(f"3. Most Improved Dataset: {best_dataset} ({best_improvement:+.2f}% with TS)\n")

            # Efficiency
            efficiency = df_success['score'] / df_success['fit_time']
            best_efficient_config = df_success.groupby('config').apply(
                lambda x: (x['score'] / x['fit_time']).mean()
            ).idxmax()
            f.write(f"4. Most Efficient Configuration: {best_efficient_config}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ Summary report saved to {report_path}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(df)}")
        print(f"Mean Score: {df_success['score'].mean():.4f}")
        print(f"Best Config: {best_config} (score={best_score:.4f})")
        if True in ts_stats.index and False in ts_stats.index:
            print(f"TS Improvement: {improvement:+.2f}%")
        print("=" * 80)


def main():
    """Main execution function"""

    print("\n" + "🚀" * 40)
    print("Starting BigFeat Time Series Testing Suite")
    print("🚀" * 40 + "\n")

    # Create tester
    tester = BigFeatTimeSeriesTester(output_dir='../bigfeat_test_results')

    # Run tests
    tester.run_comprehensive_tests()

    # Generate visualizations
    tester.generate_visualizations()

    # Generate report
    tester.generate_summary_report()

    print("\n" + "✅" * 40)
    print("Testing Complete! Check the 'bigfeat_test_results' directory for outputs.")
    print("✅" * 40 + "\n")


if __name__ == "__main__":
    main()