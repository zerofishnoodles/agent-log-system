"""
Aggregate results and create plots for redundancy analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
from pathlib import Path
from typing import List, Dict, Any

def load_results() -> List[Dict[str, Any]]:
    """Load all result files."""
    results_dir = Path("results/bird_mini_dev")
    
    # Try to load aggregated results first
    all_results_file = results_dir / "all_results.json"
    if all_results_file.exists():
        with open(all_results_file, "r") as f:
            return json.load(f)
    
    # Otherwise load individual result files
    result_files = glob.glob(str(results_dir / "*.json"))
    rows = []
    
    for file_path in result_files:
        if file_path.endswith("all_results.json"):
            continue
        
        try:
            with open(file_path, "r") as f:
                result = json.load(f)
                rows.append(result)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return rows

def create_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create pandas DataFrame from results."""
    df = pd.DataFrame(rows)
    
    if not df.empty:
        print("Redundancy Statistics:")
        print(df["redundancy"].describe())
        print(f"\nTotal questions: {len(df)}")
        print(f"Average redundancy: {df['redundancy'].mean():.3f}")
        print(f"Median redundancy: {df['redundancy'].median():.3f}")
    
    return df

def plot_total_vs_unique_by_size(rows: List[Dict[str, Any]]):
    """Plot total vs unique sub-plans by size."""
    # Collect all sizes
    all_sizes = set()
    for row in rows:
        all_sizes.update(row["by_size"].keys())
    
    sizes = sorted(all_sizes)
    
    if not sizes:
        print("No size data available for plotting")
        return
    
    # Calculate averages
    avg_totals = []
    avg_uniques = []
    
    for size in sizes:
        totals = []
        uniques = []
        
        for row in rows:
            if str(size) in row["by_size"]:
                totals.append(row["by_size"][str(size)]["total"])
                uniques.append(row["by_size"][str(size)]["unique"])
        
        if totals:
            avg_totals.append(np.mean(totals))
            avg_uniques.append(np.mean(uniques))
        else:
            avg_totals.append(0)
            avg_uniques.append(0)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, avg_totals, 'o-', label="Total", linewidth=2, markersize=6)
    plt.plot(sizes, avg_uniques, 's-', label="Distinct", linewidth=2, markersize=6)
    
    plt.xlabel("Sub-plan Size", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Total vs Distinct Sub-plans by Size", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig("results/bird_mini_dev/fig2a_repro.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot: results/fig2a_repro.png")

def plot_distinct_total_ratio_by_operator(rows: List[Dict[str, Any]]):
    """Plot distinct/total ratio by operator type."""
    # Standard operator types
    ops = ["PR", "TS", "FI", "HJ", "UA", "AG", "SO", "LI", "DI", "WI", "OT"]
    
    ratios = []
    valid_ops = []
    
    for op in ops:
        op_ratios = []
        
        for row in rows:
            if op in row["by_op"]:
                total = row["by_op"][op]["total"]
                unique = row["by_op"][op]["unique"]
                if total > 0:
                    op_ratios.append(unique / total)
        
        if op_ratios:
            ratios.append(np.mean(op_ratios))
            valid_ops.append(op)
    
    if not valid_ops:
        print("No operator data available for plotting")
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(valid_ops, ratios, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Operator Type", fontsize=12)
    plt.ylabel("Distinct / Total", fontsize=12)
    plt.title("Distinct/Total Ratio by Operator Type", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig("results/bird_mini_dev/fig2b_repro.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot: results/fig2b_repro.png")

def plot_redundancy_distribution(rows: List[Dict[str, Any]]):
    """Plot distribution of redundancy ratios."""
    if not rows:
        print("No data available for redundancy distribution plot")
        return
    
    redundancies = [row["redundancy"] for row in rows]
    
    plt.figure(figsize=(10, 6))
    plt.hist(redundancies, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(np.mean(redundancies), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(redundancies):.3f}')
    plt.axvline(np.median(redundancies), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {np.median(redundancies):.3f}')
    
    plt.xlabel("Redundancy Ratio", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Redundancy Ratios", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig("results/bird_mini_dev/redundancy_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot: results/bird_mini_dev/redundancy_distribution.png")

def create_summary_report(rows: List[Dict[str, Any]]):
    """Create a summary report of the analysis."""
    if not rows:
        print("No data available for summary report")
        return
    
    report = []
    report.append("# Redundancy Analysis Summary Report")
    report.append("=" * 50)
    report.append("")
    
    # Basic statistics
    redundancies = [row["redundancy"] for row in rows]
    totals = [row["total"] for row in rows]
    uniques = [row["unique"] for row in rows]
    
    report.append("## Overall Statistics")
    report.append(f"- Number of questions analyzed: {len(rows)}")
    report.append(f"- Average redundancy ratio: {np.mean(redundancies):.3f}")
    report.append(f"- Median redundancy ratio: {np.median(redundancies):.3f}")
    report.append(f"- Min redundancy ratio: {np.min(redundancies):.3f}")
    report.append(f"- Max redundancy ratio: {np.max(redundancies):.3f}")
    report.append(f"- Standard deviation: {np.std(redundancies):.3f}")
    report.append("")
    
    report.append("## Sub-plan Statistics")
    report.append(f"- Average total sub-plans per question: {np.mean(totals):.1f}")
    report.append(f"- Average unique sub-plans per question: {np.mean(uniques):.1f}")
    report.append(f"- Average distinct fraction: {np.mean(uniques) / np.mean(totals):.3f}")
    report.append("")
    
    # Operator breakdown
    report.append("## Operator Type Breakdown")
    all_ops = set()
    for row in rows:
        all_ops.update(row["by_op"].keys())
    
    for op in sorted(all_ops):
        op_totals = []
        op_uniques = []
        for row in rows:
            if op in row["by_op"]:
                op_totals.append(row["by_op"][op]["total"])
                op_uniques.append(row["by_op"][op]["unique"])
        
        if op_totals:
            avg_total = np.mean(op_totals)
            avg_unique = np.mean(op_uniques)
            avg_ratio = avg_unique / avg_total if avg_total > 0 else 0
            report.append(f"- {op}: {avg_total:.1f} total, {avg_unique:.1f} unique, {avg_ratio:.3f} distinct ratio")
    
    report.append("")
    report.append("## Expected vs Actual Results")
    report.append("- Expected redundancy ratio: 0.7-0.9")
    report.append(f"- Actual average redundancy: {np.mean(redundancies):.3f}")
    
    if np.mean(redundancies) >= 0.7:
        report.append("✓ Redundancy ratio meets expected range")
    else:
        report.append("✗ Redundancy ratio below expected range")
    
    report.append("")
    report.append("- Expected distinct fraction: <0.2")
    actual_distinct_fraction = np.mean(uniques) / np.mean(totals)
    report.append(f"- Actual distinct fraction: {actual_distinct_fraction:.3f}")
    
    if actual_distinct_fraction < 0.2:
        report.append("✓ Distinct fraction meets expected range")
    else:
        report.append("✗ Distinct fraction above expected range")
    
    # Save report
    with open("results/bird_mini_dev/summary_report.md", "w") as f:
        f.write("\n".join(report))
    
    print("Saved summary report: results/bird_mini_dev/summary_report.md")

def main():
    """Main function to aggregate results and create plots."""
    print("Loading results...")
    rows = load_results()
    
    if not rows:
        print("No results found. Please run compute_metrics.py first.")
        return
    
    print(f"Loaded {len(rows)} result files")
    
    # Create DataFrame
    df = create_dataframe(rows)
    
    # Create plots
    print("\nCreating plots...")
    plot_total_vs_unique_by_size(rows)
    plot_distinct_total_ratio_by_operator(rows)
    plot_redundancy_distribution(rows)
    
    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(rows)
    
    print("\nAnalysis complete! Check the results/ directory for outputs.")

if __name__ == "__main__":
    main()
