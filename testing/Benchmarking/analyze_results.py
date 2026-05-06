"""
BigFeat Benchmark Results Analysis (Publication-Quality)
Analyze and visualize results with advanced metrics:
- Frequency Stratification (H/D/W/M/Q/Y)
- Critical Difference (CD) Diagrams
- Feature Discovery Heatmaps
- Pareto Efficiency Frontier
- Detector Sensitivity & Failure Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy.stats import wilcoxon, rankdata
from datetime import datetime

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

class BenchmarkAnalyzer:
    """Analyze BigFeat benchmark results with enhanced visualizations."""

    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "benchmark_summary.csv"

        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")

        print("Loading summary results...")
        self.df = pd.read_csv(self.summary_file)
        
        # Load enhanced metrics from JSONs
        print("Loading detailed metrics from JSON logs...")
        self._enrich_data_from_jsons()

        # Configuration names
        self.configs = [
            'auto_ensemble', 
            'yes_dft', 'yes_acf', 'yes_lomb_scargle', 
            'no_standard'
        ]
        
        # Define frequency mapping for clearer labels
        self.freq_map = {
            'H': 'Hourly', 'D': 'Daily', 'W': 'Weekly', 
            'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'
        }
        # Normalize freq for plotting
        if 'freq' in self.df.columns:
            self.df['freq'] = self.df['freq'].str.upper()
            self.df['freq_label'] = self.df['freq'].map(self.freq_map).fillna('Other')
            
        print(f"Loaded results for {len(self.df)} datasets")

    def _enrich_data_from_jsons(self):
        """
        Iterate over JSON result files to extract deep metrics not in CSV:
        - Diversity Metrics (Autoregressive, Seasonal, etc.)
        - Detector Confidence Details
        """
        diversity_rows = []
        
        for _, row in self.df.iterrows():
            dataset = row['dataset']
            json_path = self.results_dir / f"{dataset}_results.json"
            
            if not json_path.exists():
                continue
                
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract diversity metrics for BigFeat configs
                for key, val in data.items():
                    if key.startswith('bigfeat_') and isinstance(val, dict):
                        config = key.replace('bigfeat_', '')
                        
                        # Get diversity metrics
                        div = val.get('diversity_metrics', {})
                        if div:
                            diversity_rows.append({
                                'dataset': dataset,
                                'config': config,
                                'diversity_total_ops': div.get('total_ops', 0),
                                'diversity_autoregressive': div.get('autoregressive', 0),
                                'diversity_volatility': div.get('volatility', 0),
                                'diversity_trend': div.get('trend', 0),
                                'diversity_seasonal': div.get('seasonal', 0),
                                'diversity_complex': div.get('complex', 0)
                            })
                            
            except Exception as e:
                print(f"Warning: Failed to parse {json_path}: {e}")
                
        # Create DataFrame from new metrics
        if diversity_rows:
            self.diversity_df = pd.DataFrame(diversity_rows)
            # Merge back into main DF? Or keep separate for specific plots?
            # Keeping separate is cleaner for long-form analysis (heatmap)
        else:
            self.diversity_df = pd.DataFrame()
            print("Warning: No diversity metrics found in JSONs (older run?)")

    # =========================================================================
    # 1. FREQUENCY-STRATIFIED ANALYSIS
    # =========================================================================
    
    def plot_faceted_boxplots(self, save_path=None):
        """Plot MASE distribution faceted by frequency."""
        print("\nGenering Frequency-Stratified Boxplots...")
        
        # Configs to compare
        methods = ['baseline', 'tsfresh', 'openfe'] + self.configs
        
        # Melt DF for boxplot
        plot_data = []
        for method in methods:
            col = f'{method}_mase'
            if col in self.df.columns:
                temp_df = self.df[['dataset', 'freq_label', col]].copy()
                temp_df.columns = ['dataset', 'Frequency', 'MASE']
                # Clean up labels for plot
                label = method.replace('yes_', '').replace('no_', '').replace('auto_', 'Auto ').replace('_', ' ').title()
                if method == 'baseline': label = 'Baseline'
                if method == 'tsfresh': label = 'TSFresh'
                if method == 'openfe': label = 'OpenFE'
                temp_df['Method'] = label
                plot_data.append(temp_df)
                
        if not plot_data:
            return
            
        plot_df = pd.concat(plot_data).reset_index(drop=True)
        
        # Define order: H -> Y
        order = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
        plot_df['Frequency'] = pd.Categorical(plot_df['Frequency'], categories=order, ordered=True)
        
        # Expanded palette
        palette = {
            'Baseline': '#95a5a6',      # Gray
            'TSFresh': '#e74c3c',       # Red
            'OpenFE': '#2ecc71',        # Green
            'Auto Ensemble': '#3498db', # Blue
            'Dft': '#5dade2',           # Light Blue
            'Acf': '#1abc9c',           # Teal
            'Lomb Scargle': '#8e44ad',  # Purple
            'Standard': '#f39c12'       # Orange
        }

        g = sns.catplot(
            data=plot_df, x='Method', y='MASE', col='Frequency', 
            kind='box', col_wrap=3, height=4, aspect=1.2,
            hue='Method', palette=palette,
            showfliers=False, legend=False
        )
        
        # Rotate x-labels to avoid overlap
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        
        g.fig.suptitle('MASE Performance by Frequency', y=1.02, fontweight='bold')
        
        out_path = save_path or (self.results_dir / 'frequency_stratified_performance.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Saved faceted plot to {out_path} and PDF")

    def frequency_win_rates(self):
        """Calculate win rates vs baseline per frequency."""
        print("\nFrequency-Specific Win Rates (Auto Ensemble vs Baseline):")
        print("-" * 60)
        
        if 'auto_ensemble_mase' not in self.df.columns:
            return

        cols = ['dataset', 'freq_label', 'baseline_mase', 'auto_ensemble_mase']
        data = self.df[cols].dropna()
        
        results = []
        for freq in ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']:
            subset = data[data['freq_label'] == freq]
            if len(subset) == 0:
                continue
                
            wins = (subset['auto_ensemble_mase'] < subset['baseline_mase']).sum()
            total = len(subset)
            rate = (wins / total) * 100
            
            # Avg improvement
            improv = (subset['baseline_mase'] - subset['auto_ensemble_mase']) / subset['baseline_mase'] * 100
            
            results.append({
                'Frequency': freq,
                'N': total,
                'Win Rate': f"{rate:.1f}%",
                'Avg Improv': f"{improv.mean():.1f}%"
            })
            
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        return res_df

    # =========================================================================
    # 2. CRITICAL DIFFERENCE (CD) DIAGRAMS
    # =========================================================================

    def plot_cd_diagram(self, save_path=None):
        """
        Plot Critical Difference (CD) Diagram using Nemenyi test.
        Visualizes statistical significance of rankings.
        """
        print("\nGenerating Critical Difference (CD) Diagram...")
        
        # Prepare ranking data
        expected_methods = ['baseline', 'tsfresh', 'openfe'] + self.configs
        
        # Create a DataFrame for ranking with ALL expected methods
        rank_data = {}
        for m in expected_methods:
             col = f'{m}_mase'
             if col in self.df.columns:
                 rank_data[col] = self.df[col]
             else:
                 # If column missing entirely, fill with infinity (worst rank)
                 rank_data[col] = pd.Series([np.inf] * len(self.df), index=self.df.index)
                 
        rank_df = pd.DataFrame(rank_data)
        
        # Also fill any individual NaNs (failed runs) with infinity
        rank_df = rank_df.fillna(np.inf)
        
        # Rank data (lower MASE = rank 1)
        # methods with np.inf will get the average rank of the "worst" positions
        ranks = rank_df.rank(axis=1, ascending=True)
        avg_ranks = ranks.mean()
        
        # Filter out methods that failed on ALL datasets (optional, but requested to show them as last)
        # If a method is all Inf, it will have max rank. We KEEP it.
        
        # Start validation for CD calculation
        # Identify valid columns (columns that actully existed + ones we filled)
        valid_cols = rank_df.columns.tolist()
        
        # Nemenyi Critical Difference Calculation
        # CD = q_alpha * sqrt(k(k+1)/(6N))
        # q_alpha for alpha=0.05 (two-tailed) 
        # Source: Demšar (2006)
        n_datasets = len(self.df)
        k = len(valid_cols)
        
        # Lookup q_alpha for infinite df (approx)
        q_alpha_lookup = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 
            6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
        }
        q_val = q_alpha_lookup.get(k, 3.2) # fallback
        
        cd = q_val * np.sqrt((k * (k + 1)) / (6 * n_datasets))
        
        print(f"  N={n_datasets}, k={k}, CD={cd:.4f}")

        # Plotting logic (simplified CD diagram)
        # Sort methods by rank
        sorted_ranks = avg_ranks.sort_values()
        labels = [c.replace('_mase', '').replace('bigfeat_', '').replace('yes_', 'BF-').replace('auto_', 'BF-Auto-') for c in sorted_ranks.index]
        values = sorted_ranks.values

        plt.figure(figsize=(10, 4))
        
        # Limits
        low_lim = 1
        high_lim = k
        
        # Draw axis
        plt.hlines(0, low_lim, high_lim, colors='k', linewidth=2)
        
        # Draw tick marks
        for x in range(low_lim, high_lim + 1):
            plt.vlines(x, -0.05, 0.05, colors='k')
            plt.text(x, 0.1, str(x), ha='center', va='bottom', fontsize=10)
            
        plt.text(low_lim, 0.2, 'Average Rank (Lower is Better)', ha='left', va='center', fontweight='bold')
        
        # Critical Difference Bar
        plt.hlines(0.5, low_lim, low_lim + cd, colors='r', linewidth=3)
        plt.text(low_lim + cd/2, 0.6, f'CD = {cd:.2f}', ha='center', va='bottom', color='r', fontweight='bold')
        
        # Plot methods
        # Use simple offset strategy to avoid overlap
        y_offsets = [0, -0.6, -1.2, -1.8, -2.4, -3.0, -3.6, -4.2]
        
        for i, (rank, label) in enumerate(zip(values, labels)):
            # Draw line to axis
            plt.plot([rank, rank], [0, 0], 'ko', markersize=5)
            
            # Simple text placement
            y_pos = -0.5 - (i % 4) * 0.4 # Stagger
            # Draw connecting line
            plt.plot([rank, rank], [0, y_pos], 'k-', alpha=0.3, linewidth=1)
            plt.text(rank, y_pos - 0.1, f'{label}\n{rank:.2f}', ha='center', va='top', fontsize=9, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Add Title
        plt.title('Critical Difference (CD) Diagram (Nemenyi Test, p<0.05)', pad=40, fontweight='bold')
        plt.axis('off')
        
        out_path = save_path or (self.results_dir / 'cd_diagram.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Saved CD diagram to {out_path} and PDF")

    # =========================================================================
    # 3. FEATURE DISCOVERY HEATMAP
    # =========================================================================
    
    def plot_feature_discovery(self, save_path=None):
        """Plot heatmap of discovered feature types across datasets."""
        print("\nGenerating Feature Discovery Heatmap...")
        
        if self.diversity_df.empty:
            print("Skipping heatmap: No diversity metrics found.")
            return
            
        # Focus on auto_ensemble
        df_auto = self.diversity_df[self.diversity_df['config'] == 'auto_ensemble'].copy()
        
        if df_auto.empty:
            return

        # Normalize counts to percentages per dataset
        feat_cols = ['diversity_autoregressive', 'diversity_volatility', 'diversity_trend', 'diversity_seasonal', 'diversity_complex']
        labels = ['Autoregressive', 'Volatility', 'Trend', 'Seasonal', 'Complex']
        
        # Calculate total
        df_auto['total'] = df_auto[feat_cols].sum(axis=1)
        
        heatmap_data = []
        full_labels = []
        
        # Sort by total features
        df_auto = df_auto.sort_values('total', ascending=False)
        
        # Prepare matrix
        data_matrix = df_auto[feat_cols].values
        # Row normalization (percentage of features)
        row_sums = data_matrix.sum(axis=1, keepdims=True)
        # Avoid div w zero
        row_sums[row_sums == 0] = 1
        data_norm = (data_matrix / row_sums) * 100
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(data_norm, cmap='viridis', annot=True, fmt='.0f', 
                   xticklabels=labels, yticklabels=df_auto['dataset'],
                   cbar_kws={'label': '% of Generated Features'})
        
        plt.title('Feature Discovery Profile (Auto Ensemble)', pad=20, fontweight='bold')
        plt.xlabel('Operator Category')
        plt.ylabel('Dataset')
        
        out_path = save_path or (self.results_dir / 'feature_discovery_heatmap.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Saved heatmap to {out_path} and PDF")

    # =========================================================================
    # 4. DETECTOR SENSITIVITY
    # =========================================================================
    
    def plot_detector_sensitivity(self, save_path=None):
        """Analyze detector behavior: Confidence vs Wins."""
        print("\nGenerating Detector Sensitivity Plots...")
        
        # 1. DFT vs ACF Confidence
        plt.figure(figsize=(8, 6))
        
        dft_col = 'auto_dft_ts_confidence' # Assuming individual run data exists or we use what we have
        # Wait, usually we run 'auto_ensemble'. 
        # If we have 'yes_dft' and 'yes_acf', we don't have their CONFIDENCE unless we inspect logs.
        # But 'auto_ensemble' logs the confidence of the SELECTED winner.
        
        # Let's plot Auto Ensemble confidence vs Performance Gain
        if 'auto_ensemble_ts_confidence' in self.df.columns and 'baseline_mase' in self.df.columns:
            
            df_plot = self.df.dropna(subset=['auto_ensemble_ts_confidence']).copy()
            df_plot['Improvement'] = (df_plot['baseline_mase'] - df_plot['auto_ensemble_mase']) / df_plot['baseline_mase'] * 100
            
            sns.scatterplot(data=df_plot, x='auto_ensemble_ts_confidence', y='Improvement', 
                           hue='freq_label', style='freq_label', s=100, palette='deep')
            
            plt.axhline(0, color='r', linestyle='--', alpha=0.5)
            plt.title('Detector Confidence vs. Performance Improvement')
            plt.xlabel('Ensemble Detector Confidence')
            plt.ylabel('MASE Improvement (%)')
            
            out_path = save_path or (self.results_dir / 'detector_confidence_impact.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            print(f"✓ Saved sensitivity plot to {out_path} and PDF")

    def plot_stationarity_impact(self, save_path=None):
        """Analyze how Stationarity (avg_lag1) affects Trend vs Seasonal features."""
        print("\nGenerating Stationarity Impact Plot...")
        
        # We need avg_lag1 from CSV (if available) and diversity metrics from JSON
        if self.diversity_df.empty:
            print("Skipping stationarity impact: No diversity metrics.")
            return
            
        # Merge diversity data with avg_lag1 from summary DF
        cols_needed = ['dataset', 'auto_ensemble_ts_avg_lag1']
        if 'auto_ensemble_ts_avg_lag1' not in self.df.columns:
            print("Skipping stationarity impact: 'auto_ensemble_ts_avg_lag1' not in CSV.")
            return
            
        df_merged = pd.merge(self.df[cols_needed], 
                            self.diversity_df[self.diversity_df['config'] == 'auto_ensemble'],
                            on='dataset')
                            
        if df_merged.empty:
            return
            
        # Calculate Trend/Seasonal Ratio
        # (Trend + Volatility) vs (Seasonal)
        # Or just Trend %
        df_merged['total'] = df_merged[['diversity_trend', 'diversity_seasonal', 'diversity_volatility', 'diversity_autoregressive', 'diversity_complex']].sum(axis=1)
        df_merged['trend_pct'] = (df_merged['diversity_trend'] / df_merged['total']) * 100
        df_merged['seasonal_pct'] = (df_merged['diversity_seasonal'] / df_merged['total']) * 100
        
        plt.figure(figsize=(10, 6))
        
        # Plot Trend % vs Lag1
        sns.regplot(data=df_merged, x='auto_ensemble_ts_avg_lag1', y='trend_pct', 
                   label='Trend Features', scatter_kws={'alpha':0.6}, color='#e67e22')
                   
        # Plot Seasonal % vs Lag1
        sns.regplot(data=df_merged, x='auto_ensemble_ts_avg_lag1', y='seasonal_pct', 
                   label='Seasonal Features', scatter_kws={'alpha':0.6}, color='#3498db')
        
        plt.title('Impact of Stationarity (Lag-1 Autocorrelation) on Feature Discovery')
        plt.xlabel('Average Lag-1 Autocorrelation (High = Non-Stationary)')
        plt.ylabel('% of Generated Features')
        plt.legend()
        
        out_path = save_path or (self.results_dir / 'stationarity_impact.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Saved stationarity impact plot to {out_path} and PDF")

    # =========================================================================
    # 5. EFFICIENCY FRONTIER (PARETO)
    # =========================================================================
    
    def plot_efficiency_frontier(self, save_path=None):
        """Identify and plot Pareto Optimal configurations."""
        print("\nGenerating Efficiency Frontier...")
        
        methods = self.configs
        avg_mase = []
        avg_time = []
        labels = []
        
        for m in methods:
            m_mase = f'{m}_mase'
            m_time = f'{m}_time'
            if m_mase in self.df.columns:
                avg_mase.append(self.df[m_mase].mean())
                avg_time.append(self.df[m_time].mean())
                labels.append(m)
        
        # Add baseline & tsfresh
        if 'baseline_mase' in self.df.columns:
            avg_mase.append(self.df['baseline_mase'].mean())
            avg_time.append(self.df['baseline_time'].mean())
            labels.append('baseline')
            
        if 'tsfresh_mase' in self.df.columns:
            avg_mase.append(self.df['tsfresh_mase'].mean())
            avg_time.append(self.df['tsfresh_time'].mean())
            labels.append('tsfresh')

        if 'openfe_mase' in self.df.columns:
            avg_mase.append(self.df['openfe_mase'].mean())
            avg_time.append(self.df['openfe_time'].mean())
            labels.append('openfe')
            
        # Identify Pareto Frontier
        # A point (t, m) dominates (t', m') if t <= t' AND m <= m' AND (t < t' OR m < m')
        # We want to MINIMIZE Time and MINIMIZE MASE.
        points = sorted(zip(avg_time, avg_mase, labels))
        pareto = []
        
        current_min_mase = float('inf')
        for t, m, l in points:
            if m < current_min_mase:
                pareto.append((t, m, l))
                current_min_mase = m
                
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Plot all points
        plt.scatter(avg_time, avg_mase, c='grey', s=50, alpha=0.6, label='Sub-optimal')
        
        # Highlight Pareto
        px, py, pl = zip(*pareto)
        plt.plot(px, py, 'b--', alpha=0.5)
        plt.scatter(px, py, c='#2ecc71', s=150, zorder=5, label='Pareto Optimal')
        
        # Annotate
        for t, m, l in zip(avg_time, avg_mase, labels):
            weight = 'bold' if l in pl else 'normal'
            plt.annotate(l.replace('bigfeat_', '').replace('auto_', 'Auto-').replace('yes_', ''), 
                        (t, m), xytext=(5, 5), textcoords='offset points', fontweight=weight)
            
        plt.title('Efficiency Frontier (Pareto Analysis)', fontweight='bold')
        plt.xlabel('Average Runtime (s) [Lower is Better]')
        plt.ylabel('Average MASE [Lower is Better]')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        out_path = save_path or (self.results_dir / 'efficiency_frontier.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Saved efficiency frontier to {out_path} and PDF")

    # =========================================================================
    # 6. SCALABILITY & FAILURE ANALYSIS
    # =========================================================================
    
    def plot_scalability(self, save_path=None):
        """Plot Runtime vs Total Timesteps."""
        print("\nGenerating Scalability Plot...")
        
        method = 'auto_ensemble'
        time_col = f'{method}_bf_time' # Feature Eng time only
        
        if time_col not in self.df.columns or 'n_series' not in self.df.columns:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Filter failures (time=0 or null)
        df_valid = self.df[self.df[time_col] > 0].copy()
        
        # Calculate features generated (total work = series * timesteps? No, BigFeat scales with n_series mostly)
        # Let's plot vs n_series * pred_length (total prediction points) or just n_series.
        # Ideally total_timesteps if available.
        
        # We need to estimate total timesteps if not provided
        # metadata usually has 'total_timesteps' if we loaded JSONs... 
        # But CSV has 'n_series' and 'pred_length'.
        # Let's use 'n_series' as primary scaling factor for now (Series-Independence).
        
        sns.scatterplot(data=df_valid, x='n_series', y=time_col, 
                       hue='freq_label', size='pred_length', sizes=(20, 200),
                       alpha=0.7, palette='viridis')
        
        # Fit linear or log trend
        x = df_valid['n_series']
        y = df_valid[time_col]
        
        # Plot y=x reference (Linear scaling)
        # plt.plot([x.min(), x.max()], [y.min(), y.max()], 'k--', alpha=0.3, label='Linear Reference')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Scalability: {method} Runtime vs Dataset Size')
        plt.xlabel('Number of Series (Log Scale)')
        plt.ylabel('Feature Engineering Time (s) (Log Scale)')
        
        out_path = save_path or (self.results_dir / 'scalability.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"✓ Saved scalability plot to {out_path} and PDF")

    def analyze_failures(self):
        """Identify and report failure modes (Auto worse than baseline or standard)."""
        print("\nFAILURE MODE ANALYSIS")
        print("="*80)
        
        if 'auto_ensemble_mase' not in self.df.columns:
            return

        # 1. Regressions vs Baseline (> 10% degradation)
        regressions = self.df[self.df['auto_ensemble_mase'] > self.df['baseline_mase'] * 1.1]
        
        print(f"\nSignificant Regressions (Auto > Baseline + 10%)")
        print("-" * 60)
        
        if not regressions.empty:
            cols = ['dataset', 'freq_label', 'baseline_mase', 'auto_ensemble_mase']
            if 'auto_ensemble_ts_confidence' in self.df.columns:
                cols.append('auto_ensemble_ts_confidence')
                
            print(regressions[cols].to_string(index=False))
            
            # Correlation check
            if 'auto_ensemble_ts_confidence' in self.df.columns:
                avg_conf_fail = regressions['auto_ensemble_ts_confidence'].mean()
                print(f"\nAverage Confidence in Failing Datasets: {avg_conf_fail:.2f}")
        else:
            print("None found.")

    # =========================================================================
    # MAIN REPORT GENERATION
    # =========================================================================

    def summary_statistics(self):
        """Print standard summary statistics."""
        # Using the base logic but keeping it simple for now
        print("\nSUMMARY MASE STATISTICS")
        print("-" * 60)
        
        methods = ['baseline', 'tsfresh', 'openfe'] + self.configs
        stats = []
        for m in methods:
            col = f'{m}_mase'
            if col in self.df.columns:
                vals = self.df[col].dropna()
                stats.append({
                    'Method': m,
                    'Mean': vals.mean(),
                    'Median': vals.median(),
                    'Std': vals.std()
                })
        print(pd.DataFrame(stats).to_string(index=False))

    def export_report(self):
        """Export comprehensive analysis report to text file."""
        report_path = self.results_dir / "analysis_report.txt"
        print(f"\nExporting analysis report to {report_path}...")
        
        with open(report_path, "w") as f:
            f.write("BIGFEAT BENCHMARK ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. Summary Statistics
            f.write("1. SUMMARY STATISTICS (MASE)\n")
            f.write("-" * 60 + "\n")
            methods = ['baseline', 'tsfresh', 'openfe'] + self.configs
            stats = []
            for m in methods:
                col = f'{m}_mase'
                if col in self.df.columns:
                    vals = self.df[col].dropna()
                    stats.append({
                        'Method': m,
                        'Mean': vals.mean(),
                        'Median': vals.median(),
                        'Std': vals.std()
                    })
            if stats:
                f.write(pd.DataFrame(stats).to_string(index=False))
            f.write("\n\n")
            
            # 2. Frequency Win Rates
            f.write("2. FREQUENCY-SPECIFIC WIN RATES (Auto vs Baseline)\n")
            f.write("-" * 60 + "\n")
            win_df = self.frequency_win_rates()
            if win_df is not None and not win_df.empty:
               f.write(win_df.to_string(index=False))
            f.write("\n\n")
            
            # 3. Failure Analysis
            f.write("3. FAILURE MODE ANALYSIS (Auto > Baseline + 10%)\n")
            f.write("-" * 60 + "\n")
            if 'auto_ensemble_mase' in self.df.columns:
                regressions = self.df[self.df['auto_ensemble_mase'] > self.df['baseline_mase'] * 1.1]
                if not regressions.empty:
                    cols = ['dataset', 'freq_label', 'baseline_mase', 'auto_ensemble_mase']
                    if 'auto_ensemble_ts_confidence' in self.df.columns:
                        cols.append('auto_ensemble_ts_confidence')
                    f.write(regressions[cols].to_string(index=False))
                    
                    if 'auto_ensemble_ts_confidence' in self.df.columns:
                        avg_conf = regressions['auto_ensemble_ts_confidence'].mean()
                        f.write(f"\n\nAverage Confidence in Failing Datasets: {avg_conf:.2f}")
                else:
                    f.write("None found.")
            f.write("\n\n")

            f.write("="*80 + "\n")
            f.write("End of Report\n")
            
        print("✓ Report exported.")
        
    def export_rankings(self):
        """Calculate and export rankings and win counts."""
        print("\nExporting rankings...")
        methods = ['baseline', 'tsfresh', 'openfe'] + self.configs
        valid_cols = [f'{m}_mase' for m in methods if f'{m}_mase' in self.df.columns]
        
        if not valid_cols:
             return

        # 1. Average Rank
        ranks = self.df[valid_cols].rank(axis=1, ascending=True)
        avg_ranks = ranks.mean().sort_values()
        
        # Save to CSV
        avg_ranks.to_csv(self.results_dir / "average_rankings.csv", header=['Average Rank'])
        print(f"✓ Saved average rankings to {self.results_dir / 'average_rankings.csv'}")

        # 2. Best Counts
        best_counts = ranks.idxmin(axis=1).value_counts()
        best_counts.index = [c.replace('_mase', '') for c in best_counts.index]
        best_counts.to_csv(self.results_dir / "best_config_counts.csv", header=['Wins'])
        print(f"✓ Saved win counts to {self.results_dir / 'best_config_counts.csv'}")

    def analyze_compactness(self):
        """Analyze feature efficiency (Compactness Argument)."""
        # Append to report
        report_path = self.results_dir / "analysis_report.txt"
        
        with open(report_path, "a") as f:
            f.write("4. COMPACTNESS ARGUMENT (Feature Efficiency)\n")
            f.write("-" * 60 + "\n")
            
            methods = ['baseline', 'tsfresh', 'openfe'] + self.configs
            data = []
            
            for m in methods:
                 feat_col = f'{m}_n_features'
                 mase_col = f'{m}_mase'
                 
                 # Handling slightly different column names in CSV
                 if m == 'baseline': feat_col = 'baseline_n_features' # Actually it is just n_features sometimes? No, csv has baseline_n_features?
                 # Let's check CSV columns in mind... benchmark.py writes 'baseline_n_features'? 
                 # Wait, benchmark.py L1082: 'Features: {baseline_results['n_features']}' is printed.
                 # summary_df row logic?
                 # L1301: baseline logic doesn't explicitly save n_features to row? 
                 # L1302: if 'baseline' in result... 
                 # Actually, benchmark.py DOES NOT seem to export baseline n_features to summary CSV in the logic I saw earlier (L1301-1313).
                 # Verify?
                 pass 
                 
            # Re-reading benchmark.py logic:
            # L1314+: BigFeat loop exports n_features_generated -> row[f'{config_name}_n_features']
            # L1346+: tsfresh exports n_features -> row['tsfresh_n_features']
            
            # So we have feature counts for BigFeat and tsfresh. Baseline is always 1 (original series) or K features?
            # Baseline (rf/ridge) usually takes prepared features... but we generally consider raw features.
            
            for m in methods:
                 if m == 'baseline': 
                     # Baseline usually implies raw features, let's assume raw features count from metadata or similar.
                     # Actually, let's skip baseline for compactness features comparison if data missing.
                     # But we definitely want to compare Auto vs tsfresh.
                     continue
                     
                 feat_col = f'{m}_n_features'
                 mase_col = f'{m}_mase'
                 
                 if feat_col in self.df.columns and mase_col in self.df.columns:
                     avg_feat = self.df[feat_col].mean()
                     avg_mase = self.df[mase_col].mean()
                     
                     data.append({
                         'Method': m,
                         'Avg Features': f"{avg_feat:.1f}",
                         'Avg MASE': f"{avg_mase:.4f}"
                     })
                     
            if data:
                f.write(pd.DataFrame(data).to_string(index=False))
            f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("End of Report (Updated)\n")
            
        print("✓ Compactness analysis appended to report.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze BigFeat benchmark results")
    parser.add_argument('--results-dir', default='./benchmark_results', help='Results directory')
    args = parser.parse_args()

    try:
        analyzer = BenchmarkAnalyzer(args.results_dir)
        
        # 1. Standard Stats
        analyzer.summary_statistics()
        
        # 2. Stratified Analysis
        analyzer.plot_faceted_boxplots()
        analyzer.frequency_win_rates()
        
        # 3. Statistical Analysis
        analyzer.plot_cd_diagram()
        
        # 4. Feature Discovery
        analyzer.plot_feature_discovery()
        
        # 5. Sensitivity & Efficiency
        analyzer.plot_detector_sensitivity()
        analyzer.plot_stationarity_impact()
        analyzer.plot_efficiency_frontier()
        
        # 6. Scalability & Failures
        analyzer.plot_scalability()
        analyzer.analyze_failures()
        
        # 7. Export Report
        analyzer.export_report()
        analyzer.export_rankings()
        analyzer.analyze_compactness()
        
        print("\n✓ Analysis Complete! All plots saved to results directory.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()