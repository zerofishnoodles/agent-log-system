"""
Main script to run the complete redundancy analysis pipeline.
"""

import os
import sys
from pathlib import Path
import subprocess

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        else:
            print(f"✗ {description} failed")
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: Virtual environment may not be activated")
    
    # Check required directories
    required_dirs = ["data", "sqls", "plans", "results"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"Creating directory: {dir_name}")
            Path(dir_name).mkdir(exist_ok=True)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("The pipeline will use sample data instead of real LLM generation")
    
    print("Environment check complete")

def main():
    """Run the complete redundancy analysis pipeline."""
    print("BIRD Text-to-SQL Redundancy Analysis Pipeline")
    print("=" * 60)
    
    # Check environment
    check_environment()
    
    # Pipeline steps
    steps = [
        ("setup_dataset.py", "Dataset Setup"),
        ("generate_sqls.py", "SQL Generation"),
        ("extract_plans.py", "Plan Extraction"),
        ("compute_metrics.py", "Metrics Computation"),
        ("aggregate_plot.py", "Aggregation and Plotting")
    ]
    
    success_count = 0
    
    for script, description in steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\nPipeline failed at step: {description}")
            print("Please check the error messages above and fix any issues.")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    if success_count == len(steps):
        print("✓ All pipeline steps completed successfully!")
        print("\nGenerated files:")
        
        # List generated files
        results_dir = Path("results")
        if results_dir.exists():
            for file in results_dir.glob("*"):
                print(f"  - results/{file.name}")
        
        sqls_dir = Path("sqls")
        if sqls_dir.exists():
            sql_count = len(list(sqls_dir.glob("*.json")))
            print(f"  - sqls/ ({sql_count} SQL files)")
        
        plans_dir = Path("plans")
        if plans_dir.exists():
            plan_count = len(list(plans_dir.glob("*.json")))
            print(f"  - plans/ ({plan_count} plan files)")
        
        print("\nCheck the results/ directory for:")
        print("  - fig2a_repro.png: Total vs unique sub-plans by size")
        print("  - fig2b_repro.png: Distinct/total ratio by operator")
        print("  - redundancy_distribution.png: Distribution of redundancy ratios")
        print("  - summary_report.md: Detailed analysis report")
        
    else:
        print(f"✗ Pipeline completed {success_count}/{len(steps)} steps")
        print("Please check the error messages and rerun the pipeline.")

if __name__ == "__main__":
    main()
