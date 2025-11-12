"""
Plan extraction and canonicalization functions for DuckDB execution plans.
"""

import json
import re
import duckdb
from pathlib import Path
from typing import List, Dict, Any
import sqlglot

def canonical(node: Dict[str, Any]) -> str:
    """
    Canonicalize a plan node to remove identifiers and normalize structure.
    
    Args:
        node: DuckDB plan node dictionary
        
    Returns:
        Canonical string representation of the node
    """
    op = node.get("operator_type", "UNKNOWN")
    children = node.get("children", [])
    
    # Recursively canonicalize children
    canonical_children = [canonical(c) for c in children]
    
    # Create signature with sorted children for consistency
    sig = f"{op}(" + ",".join(sorted(canonical_children)) + ")"
    
    # Replace identifiers with generic placeholders
    sig = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", "X", sig)
    
    return sig

def collect_subplans(node: Dict[str, Any]) -> List[str]:
    """
    Collect all sub-plans recursively from a plan node.
    
    Args:
        node: DuckDB plan node dictionary
        
    Returns:
        List of canonical sub-plan signatures
    """
    subs = [canonical(node)]
    
    for child in node.get("children", []):
        subs += collect_subplans(child)
    
    return subs

def extract_plan_from_sql(sql: str, db_path: str) -> Dict[str, Any]:
    """
    Extract execution plan from SQL query using DuckDB.
    
    Args:
        sql: SQL query string
        db_path: Path to DuckDB database file
        
    Returns:
        Dictionary with 'plan' (plan dict or None) and 'error' (error info or None)
    """
    result = {"plan": None, "error": None}
    
    try:
        # Clean SQL
        cleaned_sql = clean_sql(sql)
        
        # Connect to DuckDB database
        con = duckdb.connect()
        
        # If it's a SQLite file, attach it first
        if db_path.endswith('.sqlite') or db_path.endswith('.db'):
            con.execute(f"ATTACH '{db_path}' AS sqlite_db")
            # Modify SQL to use the attached database
            cleaned_sql = cleaned_sql.replace('FROM ', 'FROM sqlite_db.').replace('JOIN ', 'JOIN sqlite_db.')
        
        # Get execution plan in text format - this is where execution errors occur
        try:
            result_query = con.execute(f"EXPLAIN {cleaned_sql}")
            plan_row = result_query.fetchone()
            plan_text = plan_row[1] if len(plan_row) > 1 else plan_row[0]
            
            # Parse text plan into a simple structure
            plan_dict = parse_text_plan(plan_text)
            result["plan"] = plan_dict
            
        except Exception as e:
            result["error"] = {"type": "execution_error", "message": f"Execution failed: {str(e)}"}
        
        con.close()
        return result
        
    except Exception as e:
        result["error"] = {"type": "execution_error", "message": f"Execution failed: {str(e)}"}
        return result

def parse_text_plan(plan_text: str) -> Dict[str, Any]:
    """
    Parse DuckDB text-based execution plan into a structured format.
    
    Args:
        plan_text: Text-based execution plan from DuckDB
        
    Returns:
        Structured plan dictionary
    """
    lines = plan_text.split('\n')
    
    # Find operators by looking for lines with │ characters
    operators = []
    
    for line in lines:
        if '│' in line and any(op in line for op in ['PROJECTION', 'SEQ_SCAN', 'FILTER', 'HASH_JOIN', 'UNION', 'AGGREGATE', 'ORDER', 'LIMIT', 'DISTINCT']):
            # Extract operator name
            if 'PROJECTION' in line:
                operators.append('PROJECTION')
            elif 'SEQ_SCAN' in line:
                operators.append('SEQ_SCAN')
            elif 'FILTER' in line:
                operators.append('FILTER')
            elif 'HASH_JOIN' in line:
                operators.append('HASH_JOIN')
            elif 'UNION' in line:
                operators.append('UNION_ALL')
            elif 'AGGREGATE' in line:
                operators.append('AGGREGATE')
            elif 'ORDER' in line:
                operators.append('ORDER_BY')
            elif 'LIMIT' in line:
                operators.append('LIMIT')
            elif 'DISTINCT' in line:
                operators.append('DISTINCT')
    
    if not operators:
        return {"operator_type": "UNKNOWN", "children": []}
    
    # Create root node with first operator
    root_op = operators[0]
    plan_dict = {
        "operator_type": root_op,
        "children": []
    }
    
    # Add child operators
    for child_op in operators[1:]:
        plan_dict["children"].append({
            "operator_type": child_op,
            "children": []
        })
    
    return plan_dict

def extract_plans_for_question(qid: str, db_path: str) -> Dict[str, Any]:
    """
    Extract execution plans for all SQLs of a question.
    
    Args:
        qid: Question ID
        db_path: Path to DuckDB database file
        
    Returns:
        Dictionary with 'plans' (list of plan dicts) and 'error_metrics' (error statistics)
    """
    sqls_file = Path("sqls/bird_mini_dev") / f"{qid}.json"
    # Initialize error metrics
    error_metrics = {
        "total_sqls": 0,
        "successful_extractions": 0,
        "execution_errors": 0,
        "execution_error_details": []
    }
    
    if not sqls_file.exists():
        print(f"No SQLs found for question {qid}")
        return {"plans": [], "error_metrics": error_metrics}
    
    # Load SQLs
    with open(sqls_file, "r") as f:
        sqls = json.load(f)
    
    error_metrics["total_sqls"] = len(sqls)
    plans = []
    
    for i, sql in enumerate(sqls):
        # Extract plan
        result = extract_plan_from_sql(sql, db_path)
        
        if result["plan"] is not None:
            plans.append(result["plan"])
            error_metrics["successful_extractions"] += 1
            
            # Save individual plan
            plan_file = Path("plans/bird_mini_dev") / f"{qid}_plan_{i}.json"
            with open(plan_file, "w") as f:
                json.dump(result["plan"], f, indent=2)
        else:
            # Only track execution errors
            if result["error"] and result["error"]["type"] == "execution_error":
                error_metrics["execution_errors"] += 1
                error_message = result["error"]["message"]
                
                # Track execution error details
                error_metrics["execution_error_details"].append({
                    "sql_index": i,
                    "error_message": error_message,
                    "sql_preview": sql[:100] + "..." if len(sql) > 100 else sql
                })
                
                print(f"Execution error for SQL {i} of question {qid}: {error_message}")
            else:
                # For non-execution errors, just count as successful (no error)
                error_metrics["successful_extractions"] += 1
    
    return {"plans": plans, "error_metrics": error_metrics}

def validate_sql(sql: str) -> bool:
    """
    Validate SQL syntax using sqlglot.
    
    Args:
        sql: SQL query string
        
    Returns:
        True if SQL is valid, False otherwise
    """
    try:
        sqlglot.parse(sql)
        return True
    except Exception:
        return False

def clean_sql(sql: str) -> str:
    """
    Clean and normalize SQL query.
    
    Args:
        sql: Raw SQL query string
        
    Returns:
        Cleaned SQL query
    """
    # Remove extra whitespace
    sql = sql.strip()
    
    # Remove trailing semicolon
    if sql.endswith(';'):
        sql = sql[:-1]
    
    # Ensure it starts with SELECT or WITH
    if not sql.upper().startswith(("SELECT", "WITH")):
        # Try to extract SELECT statement
        select_match = re.search(r"(SELECT.*)", sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            sql = select_match.group(1)
    
    return sql.strip()

def extract_all_plans():
    """Extract plans for all questions."""
    # Load database mapping
    plans_dir = Path("plans/bird_mini_dev")
    plans_dir.mkdir(exist_ok=True)
    for file in plans_dir.glob("*.json"):
        file.unlink()
    print(f"Cleared {len(list(plans_dir.glob('*.json')))} plans")
    data_dir = Path("data/bird_mini_dev")
    with open(data_dir / "db_map.json", "r") as f:
        db_map = json.load(f)
    
    # Load questions
    # Try to load BIRD questions first
    bird_file = data_dir / "bird_questions.json"
    if bird_file.exists():
        with open(bird_file, "r") as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} BIRD questions")
    else:
        print("No questions found. Please run setup_dataset.py first.")
        return
    
    print(f"Extracting plans for {len(questions)} questions...")
    
    # Initialize overall error metrics
    overall_metrics = {
        "total_questions": len(questions),
        "questions_processed": 0,
        "total_sqls": 0,
        "total_successful": 0,
        "total_execution_errors": 0,
        "questions_with_errors": 0,
        "execution_error_summary": []
    }
    
    for question in questions:
        qid = str(question["question_id"])  # Convert to string to match db_map keys
        db_path = db_map.get(qid)
        
        if not db_path:
            print(f"No database path found for question {qid}")
            continue
        
        print(f"Extracting plans for question {qid}...")
        result = extract_plans_for_question(qid, db_path)
        plans = result["plans"]
        error_metrics = result["error_metrics"]
        
        # Update overall metrics
        overall_metrics["questions_processed"] += 1
        overall_metrics["total_sqls"] += error_metrics["total_sqls"]
        overall_metrics["total_successful"] += error_metrics["successful_extractions"]
        overall_metrics["total_execution_errors"] += error_metrics["execution_errors"]
        
        if error_metrics["execution_errors"] > 0:
            overall_metrics["questions_with_errors"] += 1
            overall_metrics["execution_error_summary"].append({
                "question_id": qid,
                "execution_errors": error_metrics["execution_errors"],
                "total_sqls": error_metrics["total_sqls"],
                "error_rate": error_metrics["execution_errors"] / error_metrics["total_sqls"] if error_metrics["total_sqls"] > 0 else 0
            })
        
        print(f"Extracted {len(plans)} plans for question {qid} (execution errors: {error_metrics['execution_errors']})")
    
    # Print final execution error summary
    print("\n" + "="*60)
    print("EXECUTION ERROR SUMMARY")
    print("="*60)
    print(f"Total questions processed: {overall_metrics['questions_processed']}")
    print(f"Total SQLs processed: {overall_metrics['total_sqls']}")
    print(f"Total successful extractions: {overall_metrics['total_successful']}")
    print(f"Total execution errors: {overall_metrics['total_execution_errors']}")
    print(f"Questions with execution errors: {overall_metrics['questions_with_errors']}")
    
    if overall_metrics['total_sqls'] > 0:
        success_rate = (overall_metrics['total_successful'] / overall_metrics['total_sqls']) * 100
        error_rate = (overall_metrics['total_execution_errors'] / overall_metrics['total_sqls']) * 100
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Execution error rate: {error_rate:.2f}%")
    
    if overall_metrics['execution_error_summary']:
        print(f"\nQuestions with execution errors:")
        for q_summary in overall_metrics['execution_error_summary']:
            print(f"  Question {q_summary['question_id']}: {q_summary['execution_errors']}/{q_summary['total_sqls']} errors ({q_summary['error_rate']:.2%})")
    
    # # Save detailed error metrics to file
    # error_report_file = Path("execution_error_report.json")
    # with open(error_report_file, "w") as f:
    #     json.dump(overall_metrics, f, indent=2)
    # print(f"\nDetailed error report saved to: {error_report_file}")

if __name__ == "__main__":
    extract_all_plans()
