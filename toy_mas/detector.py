"""
Anomaly Detector for MAS Reliability Framework

Analyzes logs in real-time for anomalies using rule-based detection.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re


class AnomalyDetector:
    """
    Rule-based anomaly detector for Multi-Agent System logs.
    
    Detects various types of anomalies:
    - Role violations (verifier making final decisions)
    - Step repetition (same tool called with identical input)
    - Premature termination (termination before verification)
    - Ignored inputs (inputs not processed)
    """
    
    def __init__(self):
        """Initialize the anomaly detector with empty state."""
        self.tool_calls = {}  # Track tool calls for repetition detection
        self.verification_status = {}  # Track verification status per task
        self.input_history = []  # Track inputs for ignored input detection
        self.task_start_time = None
        self.max_task_duration = timedelta(minutes=5)  # Configurable timeout
    
    def detect(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect anomalies in a single log event.
        
        Args:
            event: Log event dictionary
            
        Returns:
            Anomaly dictionary if detected, None otherwise
        """
        event_type = event.get("event_type")
        
        # Initialize task timing
        if event_type == "chain_start":
            self.task_start_time = datetime.fromisoformat(event["timestamp"])
            self.verification_status[event.get("chain_id", "default")] = False
        
        # Check for role violations
        anomaly = self._detect_role_violation(event)
        if anomaly:
            return anomaly
        
        # Check for step repetition
        anomaly = self._detect_step_repetition(event)
        if anomaly:
            return anomaly
        
        # Check for premature termination
        anomaly = self._detect_premature_termination(event)
        if anomaly:
            return anomaly
        
        # Check for ignored inputs
        anomaly = self._detect_ignored_inputs(event)
        if anomaly:
            return anomaly
        
        # Check for timeout
        anomaly = self._detect_timeout(event)
        if anomaly:
            return anomaly
        
        return None
    
    def _detect_role_violation(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect role violations (e.g., verifier making final decisions)."""
        agent_id = event.get("agent_id", "").lower()
        event_type = event.get("event_type")
        
        # Check if verifier is making final decisions
        if "verifier" in agent_id and event_type == "agent_finish":
            finish_reason = event.get("finish_reason", "")
            if isinstance(finish_reason, dict):
                finish_reason = str(finish_reason)
            
            # Look for decision-making keywords
            decision_keywords = ["approved", "rejected", "final", "decision", "conclude"]
            if any(keyword in finish_reason.lower() for keyword in decision_keywords):
                return {
                    "type": "role_violation",
                    "reason": f"Verifier agent '{agent_id}' made final decision: {finish_reason}",
                    "severity": "high"
                }
        
        return None
    
    def _detect_step_repetition(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect repeated tool calls with identical inputs."""
        if event.get("event_type") != "tool_start":
            return None
        
        tool_id = event.get("tool_id", "unknown")
        tool_input = event.get("tool_input", "")
        
        # Create a key for this tool call
        call_key = f"{tool_id}:{tool_input}"
        
        if call_key in self.tool_calls:
            # Check if this is a recent repetition (within last 10 events)
            last_call_time = self.tool_calls[call_key]
            current_time = datetime.fromisoformat(event["timestamp"])
            
            if current_time - last_call_time < timedelta(minutes=1):
                return {
                    "type": "step_repetition",
                    "reason": f"Tool '{tool_id}' called with identical input '{tool_input}' within 1 minute",
                    "severity": "medium"
                }
        
        # Update the tool call record
        self.tool_calls[call_key] = datetime.fromisoformat(event["timestamp"])
        
        return None
    
    def _detect_premature_termination(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect premature termination before verification."""
        if event.get("event_type") != "chain_end":
            return None
        
        chain_id = event.get("chain_id", "default")
        
        # Check if verification was completed
        if not self.verification_status.get(chain_id, False):
            outputs = event.get("outputs", {})
            if isinstance(outputs, dict):
                output_str = str(outputs).lower()
                
                # Look for verification keywords
                verification_keywords = ["verify", "check", "validate", "approve", "reject"]
                if not any(keyword in output_str for keyword in verification_keywords):
                    return {
                        "type": "premature_termination",
                        "reason": f"Chain '{chain_id}' terminated without verification step",
                        "severity": "high"
                    }
        
        return None
    
    def _detect_ignored_inputs(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect ignored inputs (inputs not processed)."""
        if event.get("event_type") == "chain_start":
            inputs = event.get("inputs", {})
            if inputs:
                self.input_history.append({
                    "timestamp": datetime.fromisoformat(event["timestamp"]),
                    "inputs": inputs,
                    "processed": False
                })
        
        elif event.get("event_type") in ["llm_end", "tool_end"]:
            # Mark recent inputs as processed
            current_time = datetime.fromisoformat(event["timestamp"])
            for input_record in self.input_history[-5:]:  # Check last 5 inputs
                if not input_record["processed"]:
                    time_diff = current_time - input_record["timestamp"]
                    if time_diff < timedelta(minutes=2):
                        input_record["processed"] = True
        
        # Check for unprocessed inputs older than 3 minutes
        current_time = datetime.fromisoformat(event["timestamp"])
        for input_record in self.input_history:
            if not input_record["processed"]:
                time_diff = current_time - input_record["timestamp"]
                if time_diff > timedelta(minutes=3):
                    return {
                        "type": "ignored_input",
                        "reason": f"Input ignored for {time_diff}: {input_record['inputs']}",
                        "severity": "medium"
                    }
        
        return None
    
    def _detect_timeout(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect task timeout."""
        if not self.task_start_time:
            return None
        
        current_time = datetime.fromisoformat(event["timestamp"])
        duration = current_time - self.task_start_time
        
        if duration > self.max_task_duration:
            return {
                "type": "timeout",
                "reason": f"Task exceeded maximum duration of {self.max_task_duration}",
                "severity": "high"
            }
        
        return None
    
    def reset(self) -> None:
        """Reset detector state for new task."""
        self.tool_calls.clear()
        self.verification_status.clear()
        self.input_history.clear()
        self.task_start_time = None


class LLMAnomalyDetector(AnomalyDetector):
    """
    LLM-based anomaly detector (placeholder for future implementation).
    
    This would use an LLM to judge whether events are anomalous.
    """
    
    def __init__(self, llm_client=None):
        """Initialize LLM-based detector."""
        super().__init__()
        self.llm_client = llm_client
    
    def detect(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect anomalies using LLM judgment.
        
        This is a placeholder implementation that falls back to rule-based detection.
        """
        # For now, use the parent class's rule-based detection
        return super().detect(event)
        
        # Future implementation would:
        # 1. Format the event for LLM analysis
        # 2. Send to LLM with anomaly detection prompt
        # 3. Parse LLM response for anomaly classification
        # 4. Return structured anomaly information

