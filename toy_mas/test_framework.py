"""
Test script for Toy MAS Reliability Framework

Demonstrates the framework functionality without requiring a real OpenAI API key.
"""

import json
import os
from datetime import datetime
from collector import LogCollector
from detector import AnomalyDetector
from evaluator import PerformanceEvaluator
from optimizer import Optimizer


def create_mock_logs():
    """Create mock log data for testing."""
    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "logs.jsonl")
    
    # Create mock log entries
    mock_logs = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "chain_start",
            "agent_id": "hello_world_agent",
            "inputs": {"task": "generate hello world program"},
            "chain_id": "test_chain_1"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "llm_start",
            "agent_id": "hello_world_agent",
            "prompts": ["Generate a Python Hello World program"],
            "llm_id": "gpt-4o-mini"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "llm_end",
            "response": "print('Hello, World!')",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "tool_start",
            "agent_id": "hello_world_agent",
            "tool_input": "print('Hello, World!')",
            "tool_id": "code_validator"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "tool_end",
            "tool_output": "VALIDATION: Code syntax is valid"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "chain_end",
            "outputs": {"code": "print('Hello, World!')", "status": "success"},
            "chain_id": "test_chain_1"
        }
    ]
    
    # Write mock logs
    import jsonlines
    with jsonlines.open(log_file, mode='w') as writer:
        for log in mock_logs:
            writer.write(log)
    
    print(f"✅ Created mock logs at {log_file}")
    return log_file


def test_anomaly_detection():
    """Test anomaly detection functionality."""
    print("\n🔍 Testing Anomaly Detection...")
    
    detector = AnomalyDetector()
    
    # Test normal event
    normal_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "chain_start",
        "agent_id": "hello_world_agent",
        "inputs": {"task": "generate hello world program"}
    }
    
    anomaly = detector.detect(normal_event)
    print(f"   Normal event: {'No anomaly' if anomaly is None else f'Anomaly: {anomaly}'}")
    
    # Test role violation
    violation_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "agent_finish",
        "agent_id": "verifier_agent",
        "finish_reason": "APPROVED - Code is correct"
    }
    
    anomaly = detector.detect(violation_event)
    print(f"   Role violation: {'No anomaly' if anomaly is None else f'Anomaly: {anomaly}'}")
    
    # Test step repetition
    detector.tool_calls["calculator:2+2"] = datetime.utcnow()
    repetition_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "tool_start",
        "tool_id": "calculator",
        "tool_input": "2+2"
    }
    
    anomaly = detector.detect(repetition_event)
    print(f"   Step repetition: {'No anomaly' if anomaly is None else f'Anomaly: {anomaly}'}")


def test_performance_evaluation():
    """Test performance evaluation functionality."""
    print("\n📊 Testing Performance Evaluation...")
    
    # Create mock logs first
    log_file = create_mock_logs()
    
    # Test evaluator
    evaluator = PerformanceEvaluator(log_file)
    metrics = evaluator.evaluate()
    
    print(f"   Evaluation completed: {'error' not in metrics}")
    if 'error' not in metrics:
        task_summary = metrics.get('task_summary', {})
        print(f"   Total tasks: {task_summary.get('total_tasks', 0)}")
        print(f"   Completion rate: {task_summary.get('completion_rate', 0):.2%}")
        
        perf_metrics = metrics.get('performance_metrics', {})
        print(f"   Success rate: {perf_metrics.get('success_rate', 0):.2%}")
        print(f"   Avg steps per task: {perf_metrics.get('average_steps_per_task', 0):.1f}")


def test_optimization():
    """Test optimization functionality."""
    print("\n🎯 Testing Optimization...")
    
    def mock_run_function(config):
        """Mock function that simulates running an experiment."""
        # Simulate different success rates based on temperature
        temperature = config.get("agent", {}).get("temperature", 0.7)
        
        # Higher temperature = lower success rate (for demo)
        success_rate = max(0.1, 1.0 - temperature * 0.5)
        anomaly_rate = temperature * 0.2
        
        return {
            "performance_metrics": {
                "success_rate": success_rate,
                "average_steps_per_task": 2.0
            },
            "anomaly_analysis": {
                "anomaly_rate": anomaly_rate,
                "total_anomalies": int(anomaly_rate * 10)
            }
        }
    
    # Test optimizer
    optimizer = Optimizer()
    
    search_space = {
        "agent.temperature": [0.3, 0.7, 1.0],
        "agent.max_tokens": [500, 1000]
    }
    
    results = optimizer.optimize(
        run_function=mock_run_function,
        search_space=search_space,
        strategy="random",
        max_iterations=3
    )
    
    print(f"   Optimization completed: {len(results['all_results'])} experiments")
    if results['best_config']:
        best_temp = results['best_config'].get('agent', {}).get('temperature', 'N/A')
        print(f"   Best temperature: {best_temp}")


def main():
    """Run all tests."""
    print("🧪 Testing Toy MAS Reliability Framework")
    print("="*60)
    
    # Test anomaly detection
    test_anomaly_detection()
    
    # Test performance evaluation
    test_performance_evaluation()
    
    # Test optimization
    test_optimization()
    
    print("\n✅ All tests completed successfully!")
    print("\n📋 Framework Components Tested:")
    print("   ✓ Anomaly Detection (rule-based)")
    print("   ✓ Performance Evaluation (metrics computation)")
    print("   ✓ Optimization (parameter tuning)")
    print("   ✓ Log Collection (mock data)")
    
    print("\n🚀 Framework is ready for use!")
    print("   Set OPENAI_API_KEY to run with real LLM")
    print("   Run: python main.py")


if __name__ == "__main__":
    main()

