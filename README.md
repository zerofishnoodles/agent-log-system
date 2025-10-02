# Toy MAS Reliability Framework

A modular research framework for Multi-Agent System (MAS) reliability analysis built on top of LangGraph.

## Overview

This framework provides four key components for analyzing and improving the reliability of Multi-Agent Systems:

1. **Log Collector** - Captures structured JSON logs from LangGraph callbacks
2. **Anomaly Detector** - Analyzes logs in real-time for anomalies
3. **Performance Evaluator** - Computes reliability metrics from logs
4. **Optimizer** - Adjusts workflow parameters to improve reliability

## Features

- 🔍 **Real-time Anomaly Detection**: Detects role violations, step repetition, premature termination, and ignored inputs
- 📊 **Comprehensive Metrics**: Success rate, anomaly count, steps per task, timing analysis
- 🎯 **Parameter Optimization**: Grid search and random search with extensible architecture
- 📝 **Structured Logging**: JSON Lines format for reproducibility
- 🧪 **Modular Design**: Easy to extend and customize

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-log-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start

Run the Hello World experiment:

```bash
cd toy_mas
python main.py
```

This will:
- Generate a Python "Hello World" program using GPT-4o-mini
- Log all events to `logs/logs.jsonl`
- Detect anomalies in real-time
- Evaluate performance metrics
- Demonstrate parameter optimization

## Project Structure

```
toy_mas/
├── main.py                # Entry point with Hello World workflow
├── collector.py           # LogCollector (LangGraph callback)
├── detector.py            # AnomalyDetector (rule-based)
├── evaluator.py           # PerformanceEvaluator (metrics)
├── optimizer.py           # Optimizer (parameter tuning)
├── tools.py               # Simple tools (calculator, echo, etc.)
├── configs/
│   └── defaults.json      # Default configurations
└── logs/
    └── logs.jsonl         # Structured event logs
```

## Components

### Log Collector
Captures structured events from LangGraph callbacks and writes them to JSON Lines format. Each log includes:
- Timestamp
- Event type (chain_start, llm_end, tool_start, etc.)
- Agent ID/role
- Message or tool input/output
- Optional anomaly information

### Anomaly Detector
Rule-based detector that identifies:
- **Role violations**: Verifier making final decisions
- **Step repetition**: Same tool called with identical input
- **Premature termination**: Termination before verification
- **Ignored inputs**: Inputs not processed within timeout
- **Timeouts**: Tasks exceeding maximum duration

### Performance Evaluator
Computes comprehensive metrics:
- Task completion rate
- Success rate
- Average steps per task
- Anomaly count and types
- Tool usage patterns
- Timing analysis

### Optimizer
Parameter optimization with multiple strategies:
- **Grid search**: Exhaustive parameter combinations
- **Random search**: Random parameter sampling
- **Extensible**: Ready for Optuna/Ray Tune integration

## Configuration

Edit `configs/defaults.json` to customize:
- LLM model and parameters
- Workflow settings (max steps, retries)
- Prompt templates
- Optimization parameters

## Example Output

```
📊 MAS RELIABILITY EVALUATION SUMMARY
============================================================

📋 Task Summary:
   Total Tasks: 1
   Completed Tasks: 1
   Completion Rate: 100.00%

⚡ Performance Metrics:
   Success Rate: 100.00%
   Avg Steps per Task: 2.0
   Avg Task Duration: 3.2s

🚨 Anomaly Analysis:
   Total Anomalies: 0
   Anomaly Rate: 0.00%

🔧 Tool Usage:
   Total Tool Calls: 1
   Tool Usage Breakdown:
     - code_validator: 1
```

## Extensions

The framework is designed to be easily extensible:

- **Add new anomaly detection rules** in `detector.py`
- **Implement LLM-based anomaly detection** using the `LLMAnomalyDetector` class
- **Add new tools** in `tools.py`
- **Integrate advanced optimizers** (Optuna, Ray Tune) in `optimizer.py`
- **Create multi-agent workflows** by extending the StateGraph in `main.py`

## Future Enhancements

- Multi-agent workflows with role-based agents
- Multimodal task support (image captioning, etc.)
- Advanced optimization with Optuna/Ray Tune
- Real-time dashboard for monitoring
- Integration with external monitoring tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.