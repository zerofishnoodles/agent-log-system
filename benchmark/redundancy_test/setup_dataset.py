"""
Script to download and set up BIRD benchmark dataset.
"""

import os
import json
import requests
import zipfile
from pathlib import Path
import sqlite3
import duckdb
import gdown

def setup_bird_dataset():
    """Set up BIRD dataset structure and create necessary mappings."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    minidev_dir = data_dir / "bird_mini_dev"
    if not minidev_dir.exists():
        print("BIRD mini-dev dataset not found. Please ensure it's downloaded and extracted.")
        return False
    
    bird_data_dir = minidev_dir / "minidev" / "MINIDEV"
    if not bird_data_dir.exists():
        print("BIRD dataset structure not found. Please check the dataset path.")
        return False
    
    print("Setting up BIRD dataset structure...")
    
    # Load questions from mini_dev_sqlite.json
    questions_file = bird_data_dir / "mini_dev_sqlite.json"
    if not questions_file.exists():
        print(f"Questions file not found: {questions_file}")
        return False
    
    with open(questions_file, "r") as f:
        questions = json.load(f)
    
    # Load table schemas from dev_tables.json
    tables_file = bird_data_dir / "dev_tables.json"
    if not tables_file.exists():
        print(f"Tables file not found: {tables_file}")
        return False
    
    with open(tables_file, "r") as f:
        tables_data = json.load(f)
    
    # Create a mapping from db_id to table schema
    db_schemas = {}
    for table_info in tables_data:
        db_id = table_info["db_id"]
        table_names = table_info["table_names"]
        column_names = table_info["column_names_original"]
        
        # Build schema string
        schema_parts = []
        table_idx = -1
        
        for col_info in column_names:
            if col_info[0] == -1:  # Special case for "*"
                continue
            
            table_name = table_names[col_info[0]]
            col_name = col_info[1]
            
            if table_idx != col_info[0]:
                table_idx = col_info[0]
                schema_parts.append(f"\nTable {table_name}:")
            
            schema_parts.append(f"  - {col_name}")
        
        db_schemas[db_id] = "\n".join(schema_parts)
    
    # Create enhanced questions with schema information
    enhanced_questions = []
    db_map = {}
    
    for question in questions[:20]:
        db_id = question["db_id"]
        qid = question["question_id"]
        
        # Add schema to question
        enhanced_question = question.copy()
        enhanced_question["schema"] = db_schemas.get(db_id, "Schema not found")
        
        enhanced_questions.append(enhanced_question)
        
        # Map question to database file
        db_file = bird_data_dir / "dev_databases" / db_id / f"{db_id}.sqlite"
        if db_file.exists():
            db_map[str(qid)] = str(db_file)
        else:
            print(f"Warning: Database file not found for {db_id}: {db_file}")
    
    # Save enhanced questions
    questions_output_file = minidev_dir / "bird_questions.json"
    with open(questions_output_file, "w") as f:
        json.dump(enhanced_questions, f, indent=2)
    
    # Save database mapping
    db_map_file = Path(minidev_dir / "db_map.json")
    with open(db_map_file, "w") as f:
        json.dump(db_map, f, indent=2)
    
    print(f"✓ Processed {len(enhanced_questions)} questions")
    print(f"✓ Created database mapping for {len(db_map)} questions")
    print(f"✓ Saved enhanced questions to {questions_output_file}")
    print(f"✓ Saved database mapping to {db_map_file}")
    
    return True

def create_sample_dataset():
    """Create a sample dataset for testing purposes."""
    print("Creating sample dataset for testing...")
    
    minidev_dir = Path("data") / "bird_mini_dev"
    minidev_dir.mkdir(exist_ok=True)
    
    # Create sample questions
    sample_questions = [
        {
            "question_id": "sample_1",
            "question": "Find all users older than 25",
            "db_id": "sample_db",
            "schema": """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50),
    age INTEGER,
    email VARCHAR(100)
);

INSERT INTO users VALUES (1, 'Alice', 30, 'alice@example.com');
INSERT INTO users VALUES (2, 'Bob', 22, 'bob@example.com');
INSERT INTO users VALUES (3, 'Charlie', 35, 'charlie@example.com');
INSERT INTO users VALUES (4, 'Diana', 28, 'diana@example.com');
"""
        },
        {
            "question_id": "sample_2", 
            "question": "Count the number of products in each category",
            "db_id": "sample_db2",
            "schema": """
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2)
);

CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50)
);

INSERT INTO categories VALUES (1, 'Electronics');
INSERT INTO categories VALUES (2, 'Clothing');
INSERT INTO categories VALUES (3, 'Books');

INSERT INTO products VALUES (1, 'Laptop', 'Electronics', 999.99);
INSERT INTO products VALUES (2, 'T-Shirt', 'Clothing', 19.99);
INSERT INTO products VALUES (3, 'Python Book', 'Books', 39.99);
INSERT INTO products VALUES (4, 'Phone', 'Electronics', 599.99);
INSERT INTO products VALUES (5, 'Jeans', 'Clothing', 49.99);
"""
        },
        {
            "question_id": "sample_3",
            "question": "Find the average salary by department",
            "db_id": "sample_db3", 
            "schema": """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50),
    department VARCHAR(50),
    salary DECIMAL(10,2)
);

INSERT INTO employees VALUES (1, 'John', 'Engineering', 80000);
INSERT INTO employees VALUES (2, 'Jane', 'Engineering', 85000);
INSERT INTO employees VALUES (3, 'Mike', 'Sales', 60000);
INSERT INTO employees VALUES (4, 'Sarah', 'Sales', 65000);
INSERT INTO employees VALUES (5, 'Tom', 'HR', 55000);
"""
        }
    ]
    
    # Save sample questions
    with open(minidev_dir / "sample_questions.json", "w") as f:
        json.dump(sample_questions, f, indent=2)
    
    # Create DuckDB databases for each question
    db_map = {}
    
    for question in sample_questions:
        qid = question["question_id"]
        db_id = question["db_id"]
        schema = question["schema"]
        
        # Create DuckDB file
        db_path = minidev_dir / f"{db_id}.duckdb"
        con = duckdb.connect(str(db_path))
        
        # Execute schema to create tables and insert data
        # Split schema into individual statements
        statements = [stmt.strip() for stmt in schema.split(';') if stmt.strip()]
        for stmt in statements:
            try:
                con.execute(stmt)
            except Exception as e:
                print(f"Warning: Failed to execute statement: {stmt[:50]}... Error: {e}")
        con.close()
        
        db_map[qid] = str(db_path)
    
    # Save database mapping
    with open(minidev_dir / "sample_db_map.json", "w") as f:
        json.dump(db_map, f, indent=2)
    
    print(f"Created {len(sample_questions)} sample questions with databases")
    return sample_questions, db_map

if __name__ == "__main__":
    # Try to set up the BIRD dataset first
    if not setup_bird_dataset():
        print("Failed to set up BIRD dataset, creating sample dataset instead...")
        create_sample_dataset()
