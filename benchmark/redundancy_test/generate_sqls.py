"""
Script to generate SQL attempts for each question using LLM.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from llm_client import generate_sql_async
import asyncio

def render_prompt(schema: str, question: str) -> str:
    """Render the prompt template for SQL generation."""
    return f"""Given the following database schema and natural language question,
write a single executable SQL query that answers it.
Return only SQL code.

Database Schema:
{schema}

Question: {question}

SQL Query:"""

def load_questions():
    """Load questions from the dataset."""
    data_dir = Path("data", "bird_mini_dev")
    
    # Try to load BIRD questions first (enhanced with schema)
    bird_file = data_dir / "bird_questions.json"
    if bird_file.exists():
        with open(bird_file, "r") as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} BIRD questions with schema information")
        return questions
    else:
        print("No questions found. Please run setup_dataset.py first.")
        return []

async def generate_sqls_for_questions_async(questions, k=2, max_concurrent_questions=10, max_concurrent_sqls=10):
    """Generate k SQL attempts for each question asynchronously."""
    sqls_dir = Path("sqls", "bird_mini_dev")
    sqls_dir.mkdir(exist_ok=True)
    # Clear the sqls directory before generating new SQLs
    for file in sqls_dir.glob("*.json"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Failed to delete {file}: {e}")
    
    print(f"Generating {k} SQL attempts for {len(questions)} questions...")
    
    # Create semaphore to limit concurrent question processing
    question_semaphore = asyncio.Semaphore(max_concurrent_questions)
    
    async def process_question(question):
        """Process a single question asynchronously."""
        async with question_semaphore:
            qid = question["question_id"]
            schema = question["schema"]
            question_text = question["question"]
            
            # Skip if already generated
            sql_file = sqls_dir / f"{qid}.json"
            if sql_file.exists():
                print(f"Skipping {qid} - already generated")
                return
            
            # Generate prompt
            prompt = render_prompt(schema, question_text)
            
            # Generate SQLs asynchronously
            print(f"Generating SQLs for question {qid}...")
            sqls = await generate_sql_async(prompt, k=k, max_concurrent=max_concurrent_sqls)
            
            # Save SQLs
            with open(sql_file, "w") as f:
                json.dump(sqls, f, indent=2)
            
            print(f"Generated {len(sqls)} SQLs for {qid}")
    
    # Process all questions concurrently
    tasks = [process_question(question) for question in questions]
    await asyncio.gather(*tasks, return_exceptions=True)

def generate_sqls_for_questions(questions, k=2):
    """Generate k SQL attempts for each question (sync wrapper)."""
    return asyncio.run(generate_sqls_for_questions_async(questions, k))

async def main_async():
    """Main async function to generate SQLs for all questions."""
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        # print("For testing purposes, we'll create sample SQLs instead.")
        
        # Create sample SQLs for testing
        # create_sample_sqls()
        # return
    
    # Load questions
    questions = load_questions()
    if not questions:
        return
    
    # Generate SQLs asynchronously
    await generate_sqls_for_questions_async(questions, k=5)

def main():
    """Main function to generate SQLs for all questions."""
    asyncio.run(main_async())

def create_sample_sqls():
    """Create sample SQLs for testing when API key is not available."""
    print("Creating sample SQLs for testing...")
    
    sqls_dir = Path("sqls", "bird_mini_dev")
    sqls_dir.mkdir(exist_ok=True)
    
    # Load questions and database mapping
    questions = load_questions()
    
    # Load database mapping
    db_map_path = Path("data", "bird_mini_dev", "db_map.json")
    if not db_map_path.exists():
        print("Error: db_map.json not found. Please run setup_dataset.py first.")
        return
    
    with open(db_map_path, "r") as f:
        db_map = json.load(f)
    
    # Generate sample SQLs for each question
    for question in questions:
        qid = str(question["question_id"])
        question_text = question["question"]
        db_path = db_map.get(qid)
        
        # Skip if already generated
        sql_file = sqls_dir / f"{qid}.json"
        if sql_file.exists():
            print(f"Skipping {qid} - already generated")
            continue
        
        # Generate sample SQLs based on question type using actual database
        sample_sqls = generate_sample_sqls_for_question(question_text, db_path)
        
        # Save SQLs
        with open(sql_file, "w") as f:
            json.dump(sample_sqls, f, indent=2)
        print(f"Created {len(sample_sqls)} sample SQLs for {qid}")

def generate_sample_sqls_for_question(question_text: str, db_path: str = None) -> list:
    """Generate sample SQLs based on question type using actual table names."""
    import sqlite3
    
    # Get actual table names and column names from the database
    actual_tables = []
    table_columns = {}
    if db_path:
        try:
            con = sqlite3.connect(db_path)
            cursor = con.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            actual_tables = [row[0] for row in cursor.fetchall()]
            
            # Get column names for each table
            for table in actual_tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                table_columns[table] = columns
            
            con.close()
        except Exception as e:
            print(f"Warning: Could not get tables from {db_path}: {e}")
    
    # Use actual table names or fallback to generic ones
    if actual_tables:
        table1 = actual_tables[0] if len(actual_tables) > 0 else "customers"
        table2 = actual_tables[1] if len(actual_tables) > 1 else actual_tables[0]
        
        # Get first column from each table (usually the ID column)
        col1 = table_columns.get(table1, ["CustomerID"])[0] if table1 in table_columns else "CustomerID"
        col2 = table_columns.get(table2, ["GasStationID"])[0] if table2 in table_columns else "GasStationID"
    else:
        table1 = "customers"
        table2 = "transactions_1k"
        col1 = "CustomerID"
        col2 = "GasStationID"
    
    question_lower = question_text.lower()
    
    # Generate 50 sample SQLs based on question patterns using actual table and column names
    base_sqls = []
    
    if "ratio" in question_lower:
        base_sqls = [
            f"SELECT COUNT(*) FROM {table1}",
            f"SELECT COUNT(*) FROM {table2}",
            f"SELECT COUNT(*) FROM {table1} WHERE {col1} > 0",
            f"SELECT COUNT(*) FROM {table2} WHERE {col2} > 0",
            f"SELECT COUNT(*) FROM {table1} WHERE {col1} IS NOT NULL"
        ]
    elif "least" in question_lower or "minimum" in question_lower:
        base_sqls = [
            f"SELECT * FROM {table1} ORDER BY {col1} ASC LIMIT 1",
            f"SELECT * FROM {table1} WHERE {col1} = (SELECT MIN({col1}) FROM {table1})",
            f"SELECT * FROM {table2} ORDER BY {col2} ASC LIMIT 1",
            f"SELECT * FROM {table1} ORDER BY {col1} ASC",
            f"SELECT * FROM {table2} ORDER BY {col2} ASC"
        ]
    elif "count" in question_lower:
        base_sqls = [
            f"SELECT COUNT(*) FROM {table1}",
            f"SELECT COUNT({col1}) FROM {table1}",
            f"SELECT COUNT(DISTINCT {col1}) FROM {table1}",
            f"SELECT COUNT(*) FROM {table2}",
            f"SELECT COUNT({col2}) FROM {table2}"
        ]
    elif "average" in question_lower or "avg" in question_lower:
        base_sqls = [
            f"SELECT AVG({col1}) FROM {table1}",
            f"SELECT AVG({col2}) FROM {table2}",
            f"SELECT COUNT(*) FROM {table1}",
            f"SELECT COUNT(*) FROM {table2}",
            f"SELECT SUM({col1}) FROM {table1}"
        ]
    else:
        # Generic SQLs using actual table and column names
        base_sqls = [
            f"SELECT * FROM {table1}",
            f"SELECT {col1} FROM {table1}",
            f"SELECT * FROM {table2}",
            f"SELECT {col2} FROM {table2}",
            f"SELECT COUNT(*) FROM {table1}"
        ]
    
    # Repeat and vary the base SQLs to get 50 total
    sample_sqls = []
    for i in range(50):
        base_sql = base_sqls[i % len(base_sqls)]
        # Add some variation
        if i % 10 == 0:
            sample_sqls.append(base_sql + " LIMIT 10")
        elif i % 10 == 1:
            sample_sqls.append(base_sql + f" ORDER BY {col1}")
        elif i % 10 == 2:
            sample_sqls.append(base_sql.replace(table1, table2).replace(col1, col2) if table1 != table2 else base_sql)
        elif i % 10 == 3:
            sample_sqls.append(base_sql + f" WHERE {col1} IS NOT NULL")
        else:
            sample_sqls.append(base_sql)
    
    return sample_sqls

if __name__ == "__main__":
    main()
