"""
LLM client for generating SQL queries using OpenAI GPT-4o-mini.
"""

import openai
import os
from typing import List
import time
import random
import asyncio

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:30090")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")  # Changed from Coder to Instruct

# Initialize Async OpenAI client with vLLM endpoint
async_client = openai.AsyncOpenAI(
    base_url=f"{VLLM_ENDPOINT}/v1",
    api_key="dummy-key"  # vLLM doesn't require authentication
)

async def generate_single_sql(prompt: str, semaphore: asyncio.Semaphore) -> str:
    """
    Generate a single SQL query asynchronously using AsyncOpenAI.
    
    Args:
        prompt: The prompt containing database schema and natural language question
        semaphore: Semaphore to limit concurrent requests
        
    Returns:
        SQL query string
    """
    async with semaphore:  # Limit concurrent requests
        try:
            response = await async_client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a SQL expert. You MUST respond with ONLY a single executable SQL query. Do not include any explanations, comments, or code in other languages. Do not use any special tokens or formatting. Just return the raw SQL query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=1000
            )
            
            sql = response.choices[0].message.content.strip()
            
            # Clean up the SQL - remove markdown formatting if present
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            
            sql = sql.strip()
            return sql
                    
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return "SELECT 1"

async def generate_sql_async(prompt: str, k: int = 50, max_concurrent: int = 10) -> List[str]:
    """
    Generate k SQL query strings asynchronously using AsyncOpenAI.
    
    Args:
        prompt: The prompt containing database schema and natural language question
        k: Number of SQL queries to generate
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        List of SQL query strings
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all SQL generations
    tasks = [
        generate_single_sql(prompt, semaphore)
        for _ in range(k)
    ]
    
    # Execute all tasks concurrently
    sqls = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    result_sqls = []
    for i, sql in enumerate(sqls):
        if isinstance(sql, Exception):
            print(f"Error in task {i+1}: {sql}")
            result_sqls.append("SELECT 1")
        else:
            result_sqls.append(sql)
    
    return result_sqls

def generate_sql(prompt: str, k: int = 50) -> List[str]:
    """
    Generate k SQL query strings using GPT-4o-mini (async wrapper).
    
    Args:
        prompt: The prompt containing database schema and natural language question
        k: Number of SQL queries to generate
        
    Returns:
        List of SQL query strings
    """
    return asyncio.run(generate_sql_async(prompt, k))


if __name__ == "__main__":
    # Test the function
    test_prompt = """
    Database schema:
    CREATE TABLE users (id INT, name VARCHAR(50), age INT);
    
    Question: Find all users older than 25.
    """
    
    sqls = generate_sql(test_prompt, k=3)
    for i, sql in enumerate(sqls):
        print(f"SQL {i+1}: {sql}")
