"""
Quick Resource Breakdown Visualization
Shows BigFeat vs Model Training time/resources clearly
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_resource_breakdown(csv_path='./benchmark_results/benchmark_summary.csv'):
    """Create comprehensive resource breakdown visualizations."""

    # Load data
    df = pd.read_csv(csv_path)

    configs = ['auto_dft', 'auto_acf', 'auto_lomb_scargle',
               'yes_dft', 'yes_acf', 'yes_lomb_scargle', 'no_dft']

    # Create figure with multiple subplots
    fig = plt.subplots(figsize=(18, 12))
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('BigFeat Benchmark: Resource Breakdown Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # ==================== Plot 1: Time Breakdown (Stacked Bar) ====================
    ax = axes[0, 0]

    configs_present = [c for c in configs if f'{c}_bf_time' in df.columns]
    if configs_present:
        bigfeat_times = [df[f'{c}_bf_time'].mean() for c in configs_present]
        model_times = [df[f'{c}_model_time'].mean() for c in configs_present]

        x = np.arange(len(configs_present))
        width = 0.6

        p1 = ax.bar(x, bigfeat_times, width, label='Feature Engineering',
                    color='steelblue', alpha=0.8)
        p2 = ax.bar(x, model_times, width, bottom=bigfeat_times,
                    label='Model Training', color='coral', alpha=0.8)

        # Add value labels
        for i, (bf, model) in enumerate(zip(bigfeat_times, model_times)):
            total = bf + model
            ax.text(i, total + 0.3, f'{total:.1f}s', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('Time Breakdown: Feature Eng vs Model Training', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs_present, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # ==================== Plot 2: Time Percentage Breakdown ====================
    ax = axes[0, 1]

    if configs_present:
        percentages = []
        for config in configs_present:
            bf_pct = df[f'{config}_bf_time'].mean() / df[f'{config}_time'].mean() * 100
            model_pct = df[f'{config}_model_time'].mean() / df[f'{config}_time'].mean() * 100
            percentages.append([bf_pct, model_pct])

        percentages = np.array(percentages)

        x = np.arange(len(configs_present))
        p1 = ax.bar(x, percentages[:, 0], width, label='Feature Engineering',
                    color='steelblue', alpha=0.8)
        p2 = ax.bar(x, percentages[:, 1], width, bottom=percentages[:, 0],
                    label='Model Training', color='coral', alpha=0.8)

        ax.set_ylabel('Percentage of Total Time', fontsize=11, fontweight='bold')
        ax.set_title('Time Allocation (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs_present, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 100])
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # ==================== Plot 3: Memory Breakdown ====================
    ax = axes[0, 2]

    if configs_present:
        bigfeat_mem = [df[f'{c}_bf_mem_mb'].mean() for c in configs_present]
        model_mem = [df[f'{c}_model_mem_mb'].mean() for c in configs_present]

        x = np.arange(len(configs_present))
        p1 = ax.bar(x, bigfeat_mem, width, label='Feature Engineering',
                    color='green', alpha=0.7)
        p2 = ax.bar(x, model_mem, width, bottom=bigfeat_mem,
                    label='Model Training', color='orange', alpha=0.7)

        ax.set_ylabel('Memory (MB)', fontsize=11, fontweight='bold')
        ax.set_title('Memory Breakdown', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs_present, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # ==================== Plot 4: BigFeat Time vs Baseline ====================
    ax = axes[1, 0]

    if configs_present and 'baseline_time' in df.columns:
        baseline_time = df['baseline_time'].mean()
        overheads = [(df[f'{c}_bf_time'].mean() - baseline_time) for c in configs_present]

        colors = ['green' if o < 0 else 'red' for o in overheads]
        bars = ax.barh(range(len(configs_present)), overheads, color=colors, alpha=0.6)

        ax.set_xlabel('Time Overhead vs Baseline (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('BigFeat Time Overhead (Baseline = 0)', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(configs_present)))
        ax.set_yticklabels(configs_present, fontsize=9)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, overheads)):
            x_pos = val + (0.5 if val > 0 else -0.5)
            ax.text(x_pos, i, f'{val:.1f}s', va='center',
                    ha='left' if val > 0 else 'right', fontsize=8)

    # ==================== Plot 5: Model Training Impact ====================
    ax = axes[1, 1]

    if configs_present and 'baseline_model_time' in df.columns:
        baseline_model = df['baseline_model_time'].mean()
        model_times_vs_baseline = []
        labels_for_plot = []

        for config in configs_present:
            if f'{config}_model_time' in df.columns:
                model_time = df[f'{config}_model_time'].mean()
                model_times_vs_baseline.append(model_time / baseline_model)
                labels_for_plot.append(config)

        if model_times_vs_baseline:
            bars = ax.barh(range(len(labels_for_plot)), model_times_vs_baseline,
                           color=['green' if x < 1 else 'coral' for x in model_times_vs_baseline],
                           alpha=0.7)

            ax.set_xlabel('Model Training Time (vs Baseline)', fontsize=11, fontweight='bold')
            ax.set_title('Model Training Slowdown Factor', fontsize=12, fontweight='bold')
            ax.set_yticks(range(len(labels_for_plot)))
            ax.set_yticklabels(labels_for_plot, fontsize=9)
            ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, label='Baseline')
            ax.grid(axis='x', alpha=0.3)
            ax.legend(fontsize=9)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, model_times_vs_baseline)):
                ax.text(val + 0.05, i, f'{val:.2f}x', va='center', fontsize=8)

    # ==================== Plot 6: Efficiency (MASE per second) ====================
    ax = axes[1, 2]

    if configs_present and 'baseline_mase' in df.columns:
        efficiencies = []
        labels_for_eff = []

        for config in configs_present:
            mase_col = f'{config}_mase'
            time_col = f'{config}_bf_time'

            if mase_col in df.columns and time_col in df.columns:
                mask = (df[mase_col].notna() & df['baseline_mase'].notna() & df[time_col].notna())
                if mask.sum() > 0:
                    mase_improvement = (df.loc[mask, 'baseline_mase'] - df.loc[mask, mase_col]).mean()
                    avg_time = df.loc[mask, time_col].mean()

                    if avg_time > 0:
                        efficiency = mase_improvement / avg_time
                        efficiencies.append(efficiency)
                        labels_for_eff.append(config)

        if efficiencies:
            colors_eff = ['green' if e > 0 else 'red' for e in efficiencies]
            bars = ax.barh(range(len(labels_for_eff)), efficiencies, color=colors_eff, alpha=0.6)

            ax.set_xlabel('MASE Improvement per Second', fontsize=11, fontweight='bold')
            ax.set_title('BigFeat Efficiency', fontsize=12, fontweight='bold')
            ax.set_yticks(range(len(labels_for_eff)))
            ax.set_yticklabels(labels_for_eff, fontsize=9)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, efficiencies)):
                x_pos = val + (0.001 if val > 0 else -0.001)
                ax.text(x_pos, i, f'{val:.4f}', va='center',
                        ha='left' if val > 0 else 'right', fontsize=8)

    # ==================== Plot 7: CPU Usage Comparison ====================
    ax = axes[2, 0]

    if configs_present:
        methods = ['baseline'] + configs_present
        cpu_data = []
        labels_cpu = []

        for method in methods:
            cpu_col = f'{method}_cpu_pct'
            if cpu_col in df.columns:
                cpu_data.append(df[cpu_col].dropna())
                labels_cpu.append('Baseline' if method == 'baseline' else method)

        if cpu_data:
            bp = ax.boxplot(cpu_data, labels=labels_cpu, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')

            ax.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
            ax.set_title('CPU Usage Distribution', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(axis='y', alpha=0.3)

    # ==================== Plot 8: Feature Count Impact on Training ====================
    ax = axes[2, 1]

    for config in configs_present[:3]:  # Just show first 3 for clarity
        feat_col = f'{config}_n_features'
        model_time_col = f'{config}_model_time'

        if feat_col in df.columns and model_time_col in df.columns:
            mask = df[feat_col].notna() & df[model_time_col].notna()
            ax.scatter(df.loc[mask, feat_col], df.loc[mask, model_time_col],
                       alpha=0.6, s=50, label=config)

    ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model Training Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Features vs Training Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ==================== Plot 9: Summary Stats Table ====================
    ax = axes[2, 2]
    ax.axis('off')

    # Create summary table
    summary_stats = []

    if 'baseline_time' in df.columns:
        baseline_time = df['baseline_time'].mean()
        baseline_mase = df['baseline_mase'].mean()
        summary_stats.append(['Baseline', f'{baseline_time:.2f}s', f'{baseline_mase:.4f}', '-'])

    for config in configs_present[:4]:  # Top 4 configs
        if f'{config}_time' in df.columns:
            total_time = df[f'{config}_time'].mean()
            bf_time = df[f'{config}_bf_time'].mean()
            mase = df[f'{config}_mase'].mean()
            improvement = ((df['baseline_mase'].mean() - mase) / df['baseline_mase'].mean() * 100)

            summary_stats.append([
                config,
                f'{total_time:.2f}s\n(BF: {bf_time:.2f}s)',
                f'{mase:.4f}',
                f'{improvement:+.1f}%'
            ])

    if summary_stats:
        table = ax.table(cellText=summary_stats,
                         colLabels=['Config', 'Time', 'MASE', 'Improvement'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(summary_stats) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    output_path = Path(csv_path).parent / 'resource_breakdown_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved detailed resource breakdown to: {output_path}")

    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize resource breakdown')
    parser.add_argument('--csv', type=str,
                        default='./benchmark_results/benchmark_summary.csv',
                        help='Path to benchmark summary CSV')

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print("Creating Resource Breakdown Visualization")
    print(f"{'=' * 80}")
    print(f"Input: {args.csv}")

    visualize_resource_breakdown(args.csv)

    print(f"\n{'=' * 80}")
    print("✅ Visualization Complete!")
    print(f"{'=' * 80}\n")