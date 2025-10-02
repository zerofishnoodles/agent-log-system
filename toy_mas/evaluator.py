"""
Performance Evaluator for MAS Reliability Framework

Computes metrics from logs for reliability analysis.
"""

import jsonlines
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter


class PerformanceEvaluator:
    """
    Evaluates performance metrics from MAS logs.
    
    Computes metrics such as:
    - Success rate
    - Anomaly count and types
    - Average steps per task
    - Task duration
    - Tool usage patterns
    """
    
    def __init__(self, log_file: str = None):
        """
        Initialize the performance evaluator.
        
        Args:
            log_file: Path to the JSON Lines log file
        """
        if log_file is None:
            # Default to logs directory relative to this file
            log_file = os.path.join(os.path.dirname(__file__), "logs", "logs.jsonl")
        
        self.log_file = Path(log_file)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate performance metrics from logs.
        
        Returns:
            Dictionary containing computed metrics
        """
        if not self.log_file.exists():
            return {"error": f"Log file {self.log_file} not found"}
        
        # Parse logs
        logs = self._parse_logs()
        
        if not logs:
            return {"error": "No logs found"}
        
        # Compute metrics
        metrics = {
            "task_summary": self._compute_task_summary(logs),
            "anomaly_analysis": self._compute_anomaly_analysis(logs),
            "performance_metrics": self._compute_performance_metrics(logs),
            "tool_usage": self._compute_tool_usage(logs),
            "timing_analysis": self._compute_timing_analysis(logs)
        }
        
        return metrics
    
    def _parse_logs(self) -> List[Dict[str, Any]]:
        """Parse logs from JSON Lines file."""
        logs = []
        try:
            with jsonlines.open(self.log_file) as reader:
                for log in reader:
                    logs.append(log)
        except Exception as e:
            print(f"Error parsing logs: {e}")
            return []
        
        return logs
    
    def _compute_task_summary(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute basic task summary statistics."""
        chain_starts = [log for log in logs if log.get("event_type") == "chain_start"]
        chain_ends = [log for log in logs if log.get("event_type") == "chain_end"]
        
        return {
            "total_tasks": len(chain_starts),
            "completed_tasks": len(chain_ends),
            "completion_rate": len(chain_ends) / len(chain_starts) if chain_starts else 0
        }
    
    def _compute_anomaly_analysis(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze anomalies in the logs."""
        anomalies = [log for log in logs if "anomaly" in log]
        
        anomaly_types = Counter()
        anomaly_severities = Counter()
        
        for anomaly_log in anomalies:
            anomaly = anomaly_log.get("anomaly", {})
            anomaly_type = anomaly.get("type", "unknown")
            severity = anomaly.get("severity", "unknown")
            
            anomaly_types[anomaly_type] += 1
            anomaly_severities[severity] += 1
        
        return {
            "total_anomalies": len(anomalies),
            "anomaly_types": dict(anomaly_types),
            "anomaly_severities": dict(anomaly_severities),
            "anomaly_rate": len(anomalies) / len(logs) if logs else 0
        }
    
    def _compute_performance_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute performance metrics."""
        # Group logs by task (chain_id)
        tasks = defaultdict(list)
        for log in logs:
            chain_id = log.get("chain_id", "default")
            tasks[chain_id].append(log)
        
        # Compute metrics per task
        task_metrics = []
        for chain_id, task_logs in tasks.items():
            task_metrics.append(self._compute_task_metrics(task_logs))
        
        if not task_metrics:
            return {"error": "No task metrics computed"}
        
        # Aggregate metrics
        total_steps = sum(metric["steps"] for metric in task_metrics)
        total_duration = sum(metric["duration"] for metric in task_metrics)
        successful_tasks = sum(1 for metric in task_metrics if metric["success"])
        
        return {
            "average_steps_per_task": total_steps / len(task_metrics) if task_metrics else 0,
            "average_task_duration_seconds": total_duration / len(task_metrics) if task_metrics else 0,
            "success_rate": successful_tasks / len(task_metrics) if task_metrics else 0,
            "total_tasks_analyzed": len(task_metrics)
        }
    
    def _compute_task_metrics(self, task_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute metrics for a single task."""
        # Count steps (LLM calls + tool calls)
        llm_calls = len([log for log in task_logs if log.get("event_type") == "llm_start"])
        tool_calls = len([log for log in task_logs if log.get("event_type") == "tool_start"])
        steps = llm_calls + tool_calls
        
        # Compute duration
        start_time = None
        end_time = None
        
        for log in task_logs:
            if log.get("event_type") == "chain_start":
                start_time = datetime.fromisoformat(log["timestamp"])
            elif log.get("event_type") == "chain_end":
                end_time = datetime.fromisoformat(log["timestamp"])
        
        duration = 0
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
        
        # Determine success (has chain_end and no critical anomalies)
        has_completion = any(log.get("event_type") == "chain_end" for log in task_logs)
        has_critical_anomalies = any(
            log.get("anomaly", {}).get("severity") == "high" for log in task_logs
        )
        
        success = has_completion and not has_critical_anomalies
        
        return {
            "steps": steps,
            "duration": duration,
            "success": success,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls
        }
    
    def _compute_tool_usage(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tool usage patterns."""
        tool_starts = [log for log in logs if log.get("event_type") == "tool_start"]
        
        tool_counts = Counter()
        tool_input_lengths = []
        
        for log in tool_starts:
            tool_id = log.get("tool_id", "unknown")
            tool_input = log.get("tool_input", "")
            
            tool_counts[tool_id] += 1
            tool_input_lengths.append(len(tool_input))
        
        return {
            "tool_usage_counts": dict(tool_counts),
            "total_tool_calls": len(tool_starts),
            "average_input_length": sum(tool_input_lengths) / len(tool_input_lengths) if tool_input_lengths else 0
        }
    
    def _compute_timing_analysis(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing patterns."""
        if not logs:
            return {}
        
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.get("timestamp", ""))
        
        # Compute intervals between events
        intervals = []
        for i in range(1, len(sorted_logs)):
            try:
                prev_time = datetime.fromisoformat(sorted_logs[i-1]["timestamp"])
                curr_time = datetime.fromisoformat(sorted_logs[i]["timestamp"])
                interval = (curr_time - prev_time).total_seconds()
                intervals.append(interval)
            except (ValueError, KeyError):
                continue
        
        if not intervals:
            return {}
        
        return {
            "average_event_interval_seconds": sum(intervals) / len(intervals),
            "max_event_interval_seconds": max(intervals),
            "min_event_interval_seconds": min(intervals),
            "total_log_duration_seconds": intervals[-1] if intervals else 0
        }
    
    def print_summary(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Print a formatted summary of the evaluation."""
        if metrics is None:
            metrics = self.evaluate()
        
        if "error" in metrics:
            print(f"❌ Evaluation Error: {metrics['error']}")
            return
        
        print("\n" + "="*60)
        print("📊 MAS RELIABILITY EVALUATION SUMMARY")
        print("="*60)
        
        # Task Summary
        task_summary = metrics.get("task_summary", {})
        print(f"\n📋 Task Summary:")
        print(f"   Total Tasks: {task_summary.get('total_tasks', 0)}")
        print(f"   Completed Tasks: {task_summary.get('completed_tasks', 0)}")
        print(f"   Completion Rate: {task_summary.get('completion_rate', 0):.2%}")
        
        # Performance Metrics
        perf_metrics = metrics.get("performance_metrics", {})
        print(f"\n⚡ Performance Metrics:")
        print(f"   Success Rate: {perf_metrics.get('success_rate', 0):.2%}")
        print(f"   Avg Steps per Task: {perf_metrics.get('average_steps_per_task', 0):.1f}")
        print(f"   Avg Task Duration: {perf_metrics.get('average_task_duration_seconds', 0):.1f}s")
        
        # Anomaly Analysis
        anomaly_analysis = metrics.get("anomaly_analysis", {})
        print(f"\n🚨 Anomaly Analysis:")
        print(f"   Total Anomalies: {anomaly_analysis.get('total_anomalies', 0)}")
        print(f"   Anomaly Rate: {anomaly_analysis.get('anomaly_rate', 0):.2%}")
        
        anomaly_types = anomaly_analysis.get("anomaly_types", {})
        if anomaly_types:
            print(f"   Anomaly Types:")
            for anomaly_type, count in anomaly_types.items():
                print(f"     - {anomaly_type}: {count}")
        
        # Tool Usage
        tool_usage = metrics.get("tool_usage", {})
        print(f"\n🔧 Tool Usage:")
        print(f"   Total Tool Calls: {tool_usage.get('total_tool_calls', 0)}")
        
        tool_counts = tool_usage.get("tool_usage_counts", {})
        if tool_counts:
            print(f"   Tool Usage Breakdown:")
            for tool, count in tool_counts.items():
                print(f"     - {tool}: {count}")
        
        print("\n" + "="*60)
