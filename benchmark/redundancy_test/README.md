# BIRD Text-to-SQL Redundancy Analysis

This benchmark quantifies redundancy across parallel LLM SQL attempts on the BIRD text-to-SQL benchmark by analyzing DuckDB execution plans.

## Overview

The analysis generates multiple SQL queries for each question, extracts execution plans from DuckDB, and computes redundancy metrics including:

- Total sub-plans (counting duplicates)
- Distinct sub-plans (unique by structure)  
- Redundancy ratio = 1 – |unique| / |total|
- Breakdown by sub-plan size
- Breakdown by root operator type (PR, TS, FI, HJ, UA)

## Setup

1. **Create working directory and install dependencies:**
   ```bash
   cd /home/ubuntu/rui/agent-log-system/benchmark/redundancy_test
   source .venv/bin/activate
   pip install duckdb pandas sqlglot tqdm matplotlib openai
   ```

2. **Set up OpenAI API key (optional):**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   If not set, the pipeline will use sample data for testing.

## Usage

### Quick Start

Run the complete pipeline:
```bash
python run_pipeline.py
```

### Individual Steps

1. **Setup dataset:**
   ```bash
   python setup_dataset.py
   ```

2. **Generate SQL attempts:**
   ```bash
   python generate_sqls.py
   ```

3. **Extract execution plans:**
   ```bash
   python extract_plans.py
   ```

4. **Compute redundancy metrics:**
   ```bash
   python compute_metrics.py
   ```

5. **Aggregate results and create plots:**
   ```bash
   python aggregate_plot.py
   ```

## Output Files

- `sqls/{qid}.json` — 50 SQL attempts per question
- `plans/{qid}_plan*.json` — DuckDB execution plans per SQL
- `results/{qid}.json` — redundancy statistics per question
- `results/fig2a_repro.png` — total vs unique sub-plans by size
- `results/fig2b_repro.png` — distinct/total ratio by operator
- `results/redundancy_distribution.png` — distribution of redundancy ratios
- `results/summary_report.md` — detailed analysis report

## Expected Results

- **Redundancy ratio:** 0.7–0.9 (70–90% redundancy)
- **Distinct fraction:** <0.2 (unique/total < 20%)
- **Growth pattern:** Total sub-plans grow faster than unique sub-plans
- **Operator breakdown:** Highest redundancy in PR and TS nodes

## Configuration

### LLM Settings
- Model: GPT-4o-mini (configurable in `llm_client.py`)
- Temperature: 0.7
- Top-p: 0.95
- Attempts per question: 50

### Operator Mapping
The system maps DuckDB operators to standardized categories:
- PR: Projection operators
- TS: Table scan operators  
- FI: Filter operators
- HJ: Join operators
- UA: Union operators
- AG: Aggregation operators
- SO: Sort operators
- LI: Limit operators
- DI: Distinct operators
- WI: Window operators
- OT: Other operators

## Troubleshooting

1. **Missing API key:** The pipeline will use sample data if no OpenAI API key is provided.

2. **DuckDB errors:** Ensure DuckDB is properly installed and databases are accessible.

3. **Plan extraction failures:** Some SQL queries may fail to generate valid execution plans.

4. **Memory issues:** For large datasets, consider processing questions in batches.

## File Structure

```
redundancy_test/
├── data/                    # Dataset files
├── sqls/                    # Generated SQL queries
├── plans/                   # Execution plans
├── results/                 # Analysis results and plots
├── llm_client.py           # LLM interface
├── setup_dataset.py       # Dataset setup
├── generate_sqls.py        # SQL generation
├── extract_plans.py         # Plan extraction
├── compute_metrics.py      # Metrics computation
├── aggregate_plot.py       # Plotting and aggregation
└── run_pipeline.py         # Main pipeline script
```

## Validation

The benchmark validates results by checking:
- High redundancy ratio (0.7–0.9)
- Low distinct fraction (<0.2)
- Proper growth patterns
- Expected operator distributions

Successful validation indicates the benchmark reproduces the trends from Figure 2 in the referenced paper.
