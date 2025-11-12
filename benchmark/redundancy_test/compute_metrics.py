"""
Compute redundancy metrics from execution plans.
"""

import json
import glob
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
from extract_plans import collect_subplans

def size(sig: str) -> int:
    """Calculate sub-plan size based on parentheses count."""
    return sig.count("(")

def root_op(sig: str) -> str:
    """Extract root operator type from signature."""
    return sig.split("(")[0]

def map_operator_type(op: str) -> str:
    """
    Map DuckDB operator types to standardized categories.
    
    Args:
        op: DuckDB operator type
        
    Returns:
        Mapped operator type (PR, TS, FI, HJ, UA, etc.)
    """
    op_mapping = {
        # Projection operators
        "PROJECTION": "PR",
        "PROJECT": "PR",
        
        # Table scan operators
        "SEQ_SCAN": "TS",
        "INDEX_SCAN": "TS",
        "TABLE_SCAN": "TS",
        
        # Filter operators
        "FILTER": "FI",
        "WHERE": "FI",
        
        # Join operators
        "HASH_JOIN": "HJ",
        "NESTED_LOOP_JOIN": "HJ",
        "MERGE_JOIN": "HJ",
        "JOIN": "HJ",
        
        # Union operators
        "UNION_ALL": "UA",
        "UNION": "UA",
        
        # Aggregation operators
        "AGGREGATE": "AG",
        "GROUP_BY": "AG",
        
        # Sort operators
        "ORDER_BY": "SO",
        "SORT": "SO",
        
        # Limit operators
        "LIMIT": "LI",
        
        # Other operators
        "DISTINCT": "DI",
        "WINDOW": "WI"
    }
    
    return op_mapping.get(op.upper(), "OT")  # OT for Other

def compute_redundancy_metrics(qid: str) -> Dict[str, Any]:
    """
    Compute redundancy metrics for a single question.
    
    Args:
        qid: Question ID
        
    Returns:
        Dictionary containing redundancy metrics
    """
    # Load SQLs
    sqls_file = Path("sqls/bird_mini_dev") / f"{qid}.json"
    if not sqls_file.exists():
        print(f"No SQLs found for question {qid}")
        return None
    
    with open(sqls_file, "r") as f:
        sqls = json.load(f)
    
    # Load database mapping
    with open("data/bird_mini_dev/db_map.json", "r") as f:
        db_map = json.load(f)
    
    db_path = db_map.get(qid)
    if not db_path:
        print(f"No database path found for question {qid}")
        return None
    
    # Collect all sub-plans
    all_subs = []
    successful_plans = 0
    
    for i, sql in enumerate(sqls):
        try:
            # Load plan
            plan_file = Path("plans/bird_mini_dev") / f"{qid}_plan_{i}.json"
            if not plan_file.exists():
                continue
            
            with open(plan_file, "r") as f:
                plan = json.load(f)
            
            # Extract sub-plans
            subs = collect_subplans(plan)
            all_subs += subs
            successful_plans += 1
            
        except Exception as e:
            print(f"Error processing plan {i} for question {qid}: {e}")
            continue
    
    if not all_subs:
        print(f"No sub-plans found for question {qid}")
        return None
    
    # Compute basic metrics
    total = len(all_subs)
    unique_subs = list(set(all_subs))
    unique = len(unique_subs)
    redundancy = 1 - unique / total if total > 0 else 0
    
    # Breakdown by sub-plan size
    size_stats = defaultdict(lambda: {"total": 0, "unique": 0})
    
    for sub in all_subs:
        s = size(sub)
        size_stats[s]["total"] += 1
    
    for sub in unique_subs:
        s = size(sub)
        size_stats[s]["unique"] += 1
    
    # Breakdown by root operator type
    op_stats = defaultdict(lambda: {"total": 0, "unique": 0})
    
    for sub in all_subs:
        op = map_operator_type(root_op(sub))
        op_stats[op]["total"] += 1
    
    for sub in unique_subs:
        op = map_operator_type(root_op(sub))
        op_stats[op]["unique"] += 1
    
    # Create result
    result = {
        "qid": qid,
        "total": total,
        "unique": unique,
        "redundancy": redundancy,
        "successful_plans": successful_plans,
        "total_sqls": len(sqls),
        "by_size": dict(size_stats),
        "by_op": dict(op_stats)
    }
    
    return result

def compute_all_metrics():
    """Compute redundancy metrics for all questions."""
    results_dir = Path("results/bird_mini_dev")
    results_dir.mkdir(exist_ok=True)
    
    # Load questions
    data_dir = Path("data/bird_mini_dev")
    
    # Try to load BIRD questions first
    bird_file = data_dir / "bird_questions.json"
    if bird_file.exists():
        with open(bird_file, "r") as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} BIRD questions")
    else:
        print("No questions found. Please run setup_dataset.py first.")
        return
    
    print(f"Computing metrics for {len(questions)} questions...")
    
    all_results = []
    
    for question in questions:
        qid = str(question["question_id"])  # Convert to string to match plan files
        print(f"Computing metrics for question {qid}...")
        
        result = compute_redundancy_metrics(qid)
        
        if result:
            # Save individual result
            result_file = results_dir / f"{qid}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            
            all_results.append(result)
            print(f"Question {qid}: {result['total']} total, {result['unique']} unique, {result['redundancy']:.3f} redundancy")
        else:
            print(f"Failed to compute metrics for question {qid}")
    
    # Save aggregated results
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComputed metrics for {len(all_results)} questions")
    
    # Print summary statistics
    if all_results:
        redundancies = [r["redundancy"] for r in all_results]
        totals = [r["total"] for r in all_results]
        uniques = [r["unique"] for r in all_results]
        
        print(f"\nSummary Statistics:")
        print(f"Average redundancy: {sum(redundancies) / len(redundancies):.3f}")
        print(f"Min redundancy: {min(redundancies):.3f}")
        print(f"Max redundancy: {max(redundancies):.3f}")
        print(f"Average total sub-plans: {sum(totals) / len(totals):.1f}")
        print(f"Average unique sub-plans: {sum(uniques) / len(uniques):.1f}")

if __name__ == "__main__":
    compute_all_metrics()
