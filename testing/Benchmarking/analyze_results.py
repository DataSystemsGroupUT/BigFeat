"""
BigFeat Benchmark Results Analysis (Refined)
Analyze and visualize results with publication-quality outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 100})


class BenchmarkAnalyzer:
    """Analyze BigFeat benchmark results with enhanced visualizations."""

    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "benchmark_summary.csv"

        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")

        self.df = pd.read_csv(self.summary_file)

        # Configuration names
        self.configs = [
            'auto_dft', 'auto_acf', 'auto_lomb_scargle',
            'yes_dft', 'yes_acf', 'yes_lomb_scargle',
            'no_dft'
        ]

        # Handle negative memory readings (memory released)
        # Keep raw for analysis, clean for display
        self.df_display = self.df.copy()
        for col in self.df_display.columns:
            if 'mem_mb' in col:
                self.df_display[col] = self.df_display[col].apply(
                    lambda x: max(0, x) if pd.notnull(x) else x
                )

        print(f"Loaded results for {len(self.df)} datasets")
        print(f"Configurations: {len(self.configs)} + baseline")

    def summary_statistics(self):
        """Print comprehensive summary statistics."""
        output = []
        output.append("\n" + "="*80)
        output.append("SUMMARY STATISTICS")
        output.append("="*80)

        # 1. MASE comparison
        output.append("\n1. MASE Performance (Lower is Better)")
        output.append("-" * 70)
        methods = ['baseline'] + self.configs
        stats = []

        for method in methods:
            col = f'{method}_mase'
            if col in self.df.columns:
                values = self.df[col].dropna()
                if len(values) > 0:
                    stats.append({
                        'Method': method,
                        'Mean': values.mean(),
                        'Median': values.median(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'N': len(values)
                    })

        if stats:
            stats_df = pd.DataFrame(stats).sort_values('Median')
            output.append(stats_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

            # Save to CSV
            stats_df.to_csv(self.results_dir / "mase_statistics.csv", index=False)

        # 2. Win Rates
        output.append("\n\n2. Win Rates vs Baseline")
        output.append("-" * 70)
        win_stats = []

        for config in self.configs:
            mase_col = f'{config}_mase'
            if mase_col in self.df.columns:
                mask = self.df[mase_col].notna() & self.df['baseline_mase'].notna()
                wins = (self.df.loc[mask, mase_col] < self.df.loc[mask, 'baseline_mase']).sum()
                total = mask.sum()

                if total > 0:
                    win_rate = wins / total * 100

                    # Per-dataset improvements
                    improvements = (
                        (self.df.loc[mask, 'baseline_mase'] - self.df.loc[mask, mase_col]) /
                        self.df.loc[mask, 'baseline_mase'] * 100
                    )

                    win_stats.append({
                        'Config': config,
                        'Wins': f'{wins}/{total}',
                        'Win Rate %': win_rate,
                        'Avg Improv %': improvements.mean(),
                        'Median Improv %': improvements.median()
                    })

        if win_stats:
            win_df = pd.DataFrame(win_stats).sort_values('Win Rate %', ascending=False)
            output.append(win_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

            # Save to CSV
            win_df.to_csv(self.results_dir / "win_rates.csv", index=False)

        # 3. Resource Usage (Total)
        output.append("\n\n3. Resource Usage (Total Pipeline)")
        output.append("-" * 70)
        resource_stats = []

        for method in methods:
            time_col = f'{method}_time'
            cpu_col = f'{method}_cpu_pct'
            mem_col = f'{method}_mem_mb'

            if time_col in self.df.columns:
                # Use display df for memory (non-negative)
                resource_stats.append({
                    'Method': method,
                    'Time (s)': self.df[time_col].mean(),
                    'CPU %': self.df[cpu_col].mean() if cpu_col in self.df.columns else np.nan,
                    'Mem (MB)': self.df_display[mem_col].mean() if mem_col in self.df.columns else np.nan
                })

        if resource_stats:
            resource_df = pd.DataFrame(resource_stats)
            output.append(resource_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

        # 4. BigFeat-Specific Resources
        output.append("\n\n4. BigFeat Feature Engineering Overhead")
        output.append("-" * 70)
        bf_resource_stats = []

        for config in self.configs:
            bf_time_col = f'{config}_bf_time'
            bf_cpu_col = f'{config}_bf_cpu_pct'
            bf_mem_col = f'{config}_bf_mem_mb'

            if bf_time_col in self.df.columns:
                bf_resource_stats.append({
                    'Config': config,
                    'BF Time (s)': self.df[bf_time_col].mean(),
                    'BF CPU %': self.df[bf_cpu_col].mean() if bf_cpu_col in self.df.columns else np.nan,
                    'BF Mem (MB)': self.df_display[bf_mem_col].mean() if bf_mem_col in self.df.columns else np.nan,
                    'Time Overhead (s)': self.df[bf_time_col].mean() - self.df['baseline_time'].mean()
                })

        if bf_resource_stats:
            bf_df = pd.DataFrame(bf_resource_stats)
            output.append(bf_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

            # Save to CSV
            bf_df.to_csv(self.results_dir / "bigfeat_overhead.csv", index=False)

        # 5. Model Training Breakdown
        output.append("\n\n5. Model Training Resource Breakdown")
        output.append("-" * 70)
        model_stats = []

        if 'baseline_model_time' in self.df.columns:
            model_stats.append({
                'Method': 'baseline',
                'Model Time (s)': self.df['baseline_model_time'].mean(),
                'Model CPU %': self.df.get('baseline_model_cpu_pct', pd.Series([np.nan])).mean(),
                'Prep Time (s)': self.df.get('baseline_prep_time', pd.Series([0])).mean()
            })

        for config in self.configs:
            model_time_col = f'{config}_model_time'
            bf_time_col = f'{config}_bf_time'

            if model_time_col in self.df.columns:
                model_stats.append({
                    'Method': config,
                    'Model Time (s)': self.df[model_time_col].mean(),
                    'Model CPU %': self.df.get(f'{config}_model_cpu_pct', pd.Series([np.nan])).mean(),
                    'FeatEng Time (s)': self.df[bf_time_col].mean() if bf_time_col in self.df.columns else np.nan
                })

        if model_stats:
            model_df = pd.DataFrame(model_stats)
            output.append(model_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

        # 6. Time Series Detection
        output.append("\n\n6. Time Series Detection (Auto Modes)")
        output.append("-" * 70)

        for detector in ['dft', 'acf', 'lomb_scargle']:
            config = f'auto_{detector}'
            ts_col = f'{config}_ts_enabled'
            conf_col = f'{config}_ts_confidence'

            if ts_col in self.df.columns:
                enabled = self.df[ts_col].sum()
                total = self.df[ts_col].notna().sum()

                output.append(f"\n{detector.upper()}:")
                output.append(f"  Enabled: {enabled}/{total} ({enabled/total*100:.1f}%)")

                if conf_col in self.df.columns:
                    enabled_mask = self.df[ts_col] == True
                    if enabled_mask.sum() > 0:
                        avg_conf = self.df.loc[enabled_mask, conf_col].mean()
                        output.append(f"  Avg Confidence: {avg_conf:.2f}")

        # Print everything
        print("\n".join(output))
        print("\n" + "="*80 + "\n")

    def plot_performance_comparison(self, save_path: str = None):
        """Generate publication-quality comparison plots."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        methods = ['baseline'] + self.configs

        # 1. MASE Boxplot with strip overlay
        ax1 = fig.add_subplot(gs[0, 0])
        plot_data = []
        for method in methods:
            col = f'{method}_mase'
            if col in self.df.columns:
                for val in self.df[col].dropna():
                    plot_data.append({'Method': method, 'MASE': val})

        if plot_data:
            plot_df = pd.DataFrame(plot_data)

            # Color palette: baseline grey, others blue
            palette = {m: '#95a5a6' if m == 'baseline' else '#3498db' for m in methods}

            sns.boxplot(x='Method', y='MASE', data=plot_df, ax=ax1, palette=palette,
                       showfliers=False, width=0.6)
            sns.stripplot(x='Method', y='MASE', data=plot_df, ax=ax1,
                         color='black', alpha=0.3, size=3)

            ax1.set_title('MASE Distribution', fontsize=13, fontweight='bold')
            ax1.set_xlabel('')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax1.set_ylabel('MASE (Lower is Better)', fontsize=10)
            ax1.grid(axis='y', alpha=0.3)

        # 2. Win Rate Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        win_rates = []
        labels = []

        for config in self.configs:
            mase_col = f'{config}_mase'
            if mase_col in self.df.columns:
                mask = self.df[mase_col].notna() & self.df['baseline_mase'].notna()
                if mask.sum() > 0:
                    wins = (self.df.loc[mask, mase_col] < self.df.loc[mask, 'baseline_mase']).sum()
                    win_rates.append(wins / mask.sum() * 100)
                    labels.append(config)

        if win_rates:
            # Color: green if >= 50%, red otherwise
            colors = ['#27ae60' if x >= 50 else '#e74c3c' for x in win_rates]
            bars = ax2.bar(range(len(labels)), win_rates, color=colors, alpha=0.8, width=0.7)

            ax2.axhline(50, color='grey', linestyle='--', alpha=0.5, linewidth=1)
            ax2.set_title('Win Rate vs Baseline', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Win Rate (%)', fontsize=10)
            ax2.set_ylim(0, 100)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax2.grid(axis='y', alpha=0.3)

            # Add percentage labels on bars
            for i, (bar, rate) in enumerate(zip(bars, win_rates)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)

        # 3. Efficiency Scatter
        ax3 = fig.add_subplot(gs[0, 2])
        scatter_data = []

        for config in self.configs:
            mase_col = f'{config}_mase'
            time_col = f'{config}_time'

            if mase_col in self.df.columns and time_col in self.df.columns:
                mask = self.df[mase_col].notna() & self.df['baseline_mase'].notna()
                if mask.sum() > 0:
                    avg_imp = ((self.df.loc[mask, 'baseline_mase'] - self.df.loc[mask, mase_col]) /
                              self.df.loc[mask, 'baseline_mase'] * 100).mean()
                    avg_time = self.df.loc[mask, time_col].mean()

                    scatter_data.append({
                        'Config': config,
                        'Improvement': avg_imp,
                        'Time': avg_time,
                        'Mode': 'auto' if 'auto' in config else ('yes' if 'yes' in config else 'no')
                    })

        if scatter_data:
            sc_df = pd.DataFrame(scatter_data)

            # Color by mode
            mode_colors = {'auto': '#3498db', 'yes': '#2ecc71', 'no': '#e67e22'}

            for mode, color in mode_colors.items():
                mask = sc_df['Mode'] == mode
                if mask.sum() > 0:
                    ax3.scatter(sc_df.loc[mask, 'Time'], sc_df.loc[mask, 'Improvement'],
                               color=color, s=100, alpha=0.7, label=f'Mode: {mode}', edgecolors='black')

            # Baseline reference
            if 'baseline_time' in self.df.columns:
                baseline_time = self.df['baseline_time'].mean()
                ax3.scatter(baseline_time, 0, color='#95a5a6', marker='*', s=300,
                           label='Baseline', edgecolors='black', linewidths=1.5)

            ax3.axhline(0, color='grey', linestyle='-', alpha=0.3, linewidth=1)
            ax3.set_title('Efficiency Frontier', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Runtime (seconds)', fontsize=10)
            ax3.set_ylabel('MASE Improvement (%)', fontsize=10)
            ax3.legend(fontsize=8, loc='best')
            ax3.grid(alpha=0.3)

        # 4. Time Breakdown (Stacked Bar)
        ax4 = fig.add_subplot(gs[1, :2])
        configs_with_breakdown = [c for c in self.configs if f'{c}_bf_time' in self.df.columns]

        if configs_with_breakdown:
            bf_times = [self.df[f'{c}_bf_time'].mean() for c in configs_with_breakdown]
            total_times = [self.df[f'{c}_time'].mean() for c in configs_with_breakdown]
            model_times = [total - bf for total, bf in zip(total_times, bf_times)]

            x = np.arange(len(configs_with_breakdown))
            width = 0.6

            ax4.bar(x, bf_times, width, label='BigFeat', color='#3498db', alpha=0.8)
            ax4.bar(x, model_times, width, bottom=bf_times, label='Model + Overhead',
                   color='#e67e22', alpha=0.8)

            # Baseline reference line
            if 'baseline_time' in self.df.columns:
                baseline_time = self.df['baseline_time'].mean()
                ax4.axhline(baseline_time, color='#95a5a6', linestyle='--',
                           linewidth=2, label='Baseline Total', alpha=0.7)

            ax4.set_ylabel('Time (seconds)', fontsize=10)
            ax4.set_title('Time Breakdown: BigFeat vs Model Training', fontsize=13, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(configs_with_breakdown, rotation=45, ha='right', fontsize=9)
            ax4.legend(fontsize=9)
            ax4.grid(axis='y', alpha=0.3)

        # 5. Memory Breakdown
        ax5 = fig.add_subplot(gs[1, 2])

        if configs_with_breakdown:
            bf_mem = [self.df_display[f'{c}_bf_mem_mb'].mean() for c in configs_with_breakdown]
            total_mem = [self.df_display[f'{c}_mem_mb'].mean() for c in configs_with_breakdown]
            other_mem = [max(0, total - bf) for total, bf in zip(total_mem, bf_mem)]

            ax5.bar(x, bf_mem, width, label='BigFeat', color='#16a085', alpha=0.8)
            ax5.bar(x, other_mem, width, bottom=bf_mem, label='Model + Other',
                   color='#f39c12', alpha=0.8)

            # Baseline reference
            if 'baseline_mem_mb' in self.df_display.columns:
                baseline_mem = self.df_display['baseline_mem_mb'].mean()
                ax5.axhline(baseline_mem, color='#95a5a6', linestyle='--',
                           linewidth=2, label='Baseline', alpha=0.7)

            ax5.set_ylabel('Memory (MB)', fontsize=10)
            ax5.set_title('Memory Breakdown', fontsize=13, fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(configs_with_breakdown, rotation=45, ha='right', fontsize=9)
            ax5.legend(fontsize=9)
            ax5.grid(axis='y', alpha=0.3)

        # 6. Efficiency (MASE improvement per second)
        ax6 = fig.add_subplot(gs[2, :])

        if configs_with_breakdown:
            efficiencies = []
            labels_eff = []

            for config in configs_with_breakdown:
                mase_col = f'{config}_mase'
                time_col = f'{config}_bf_time'

                if mase_col in self.df.columns and time_col in self.df.columns:
                    mask = (self.df[mase_col].notna() &
                           self.df['baseline_mase'].notna() &
                           self.df[time_col].notna())

                    if mask.sum() > 0:
                        mase_improvement = (self.df.loc[mask, 'baseline_mase'] -
                                          self.df.loc[mask, mase_col]).mean()
                        avg_time = self.df.loc[mask, time_col].mean()

                        if avg_time > 0:
                            efficiency = mase_improvement / avg_time
                            efficiencies.append(efficiency)
                            labels_eff.append(config)

            if efficiencies:
                colors_eff = ['#27ae60' if e > 0 else '#e74c3c' for e in efficiencies]
                bars = ax6.barh(range(len(labels_eff)), efficiencies, color=colors_eff, alpha=0.8)

                ax6.set_xlabel('MASE Improvement per Second', fontsize=10)
                ax6.set_title('Efficiency: Performance Gain per Time Unit', fontsize=13, fontweight='bold')
                ax6.set_yticks(range(len(labels_eff)))
                ax6.set_yticklabels(labels_eff, fontsize=9)
                ax6.axvline(0, color='black', linestyle='-', linewidth=1)
                ax6.grid(axis='x', alpha=0.3)

        plt.suptitle('BigFeat Benchmark Comprehensive Analysis',
                    fontsize=16, fontweight='bold', y=0.995)

        out_path = save_path or (self.results_dir / 'comprehensive_analysis.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comprehensive plot to {out_path}")
        plt.close()

    def plot_detector_comparison(self, save_path: str = None):
        """Compare different detectors across modes."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        modes = ['auto', 'yes', 'no']
        detectors = ['dft', 'acf', 'lomb_scargle']

        for idx, mode in enumerate(modes):
            ax = axes[idx]

            if mode == 'no':
                config = 'no_dft'
                mase_col = f'{config}_mase'
                if mase_col in self.df.columns:
                    values = self.df[mase_col].dropna()
                    bp = ax.boxplot([values], labels=['dft'], patch_artist=True,
                                   widths=0.5, showfliers=False)
                    for patch in bp['boxes']:
                        patch.set_facecolor('#3498db')
            else:
                mase_data = []
                labels = []

                for detector in detectors:
                    config = f'{mode}_{detector}'
                    mase_col = f'{config}_mase'

                    if mase_col in self.df.columns:
                        values = self.df[mase_col].dropna()
                        if len(values) > 0:
                            mase_data.append(values)
                            labels.append(detector)

                if mase_data:
                    bp = ax.boxplot(mase_data, labels=labels, patch_artist=True,
                                   widths=0.5, showfliers=False)
                    colors = ['#3498db', '#2ecc71', '#e67e22']
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)

            ax.set_ylabel('MASE', fontsize=11)
            ax.set_title(f"Mode: '{mode}'", fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', labelsize=10)

        plt.suptitle('Detector Comparison Across Modes', fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = save_path or (self.results_dir / 'detector_comparison.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detector comparison to {out_path}")
        plt.close()

    def best_configuration_per_dataset(self):
        """Find best configuration for each dataset."""
        configs = ['baseline'] + self.configs
        mase_cols = [f'{c}_mase' for c in configs]

        valid_cols = [c for c in mase_cols if c in self.df.columns]

        if not valid_cols:
            return pd.DataFrame()

        self.df['best_config_col'] = self.df[valid_cols].idxmin(axis=1)
        self.df['best_config'] = self.df['best_config_col'].str.replace('_mase', '')
        self.df['best_mase'] = self.df[valid_cols].min(axis=1)

        print("\n" + "="*80)
        print("BEST CONFIGURATION WINS")
        print("="*80)
        winner_counts = self.df['best_config'].value_counts()
        print(winner_counts.to_string())

        # Save to CSV
        winner_counts.to_csv(self.results_dir / "best_config_counts.csv", header=['Count'])

        return self.df[['dataset', 'best_config', 'best_mase']]

    def export_report(self, output_file: str = None):
        """Export detailed analysis report."""
        if output_file is None:
            output_file = self.results_dir / "analysis_report.txt"

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BIGFEAT BENCHMARK ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            # Include all analysis sections
            methods = ['baseline'] + self.configs

            # MASE Summary
            f.write("1. MASE PERFORMANCE SUMMARY\n")
            f.write("-"*50 + "\n")
            for method in methods:
                col = f'{method}_mase'
                if col in self.df.columns:
                    values = self.df[col].dropna()
                    if len(values) > 0:
                        f.write(f"\n{method}:\n")
                        f.write(f"  Mean: {values.mean():.4f}\n")
                        f.write(f"  Median: {values.median():.4f}\n")
                        f.write(f"  Std: {values.std():.4f}\n")
                        f.write(f"  Range: [{values.min():.4f}, {values.max():.4f}]\n")

            # Best configs
            f.write("\n\n2. BEST CONFIGURATION PER DATASET\n")
            f.write("-"*50 + "\n")
            best_df = self.best_configuration_per_dataset()
            f.write(best_df.to_string(index=False))

            f.write("\n\n3. CONFIGURATION RANKINGS\n")
            f.write("-"*50 + "\n")
            f.write(self.df['best_config'].value_counts().to_string())

        print(f"✓ Report saved to {output_file}")


def main():
    """Main analysis script."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze BigFeat benchmark results")
    parser.add_argument('--results-dir', default='./benchmark_results', help='Results directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--report', action='store_true', help='Export detailed report')

    args = parser.parse_args()

    # Load and analyze
    analyzer = BenchmarkAnalyzer(args.results_dir)

    # Print summary statistics
    analyzer.summary_statistics()

    # Best configuration analysis
    print("\n")
    best_df = analyzer.best_configuration_per_dataset()
    print("\nTop 10 Best Configurations:")
    print(best_df.head(10).to_string(index=False))

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        analyzer.plot_performance_comparison()
        analyzer.plot_detector_comparison()

    # Export report
    if args.report:
        print("\nExporting detailed report...")
        analyzer.export_report()

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()