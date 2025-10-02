"""
Log Collector for MAS Reliability Framework

Captures structured JSON logs from LangGraph callbacks and writes them to logs.jsonl.
"""

import json
import jsonlines
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage


class LogCollector(BaseCallbackHandler):
    """
    Collects structured logs from LangGraph callbacks and writes them to JSON Lines format.
    """
    
    def __init__(self, log_file: str = None, detector: Optional[Any] = None):
        """
        Initialize the log collector.
        
        Args:
            log_file: Path to the log file (relative to project root)
            detector: Optional anomaly detector for real-time detection
        """
        if log_file is None:
            # Default to logs directory relative to this file
            log_file = os.path.join(os.path.dirname(__file__), "logs", "logs.jsonl")
        
        self.log_file = Path(log_file)
        self.detector = detector
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            pass  # Create empty file
    
    def _write_log(self, event: Dict[str, Any]) -> None:
        """Write a single log event to the JSON Lines file."""
        with jsonlines.open(self.log_file, mode='a') as writer:
            writer.write(event)
        
        # Forward to anomaly detector if available
        if self.detector:
            anomaly = self.detector.detect(event)
            if anomaly:
                print(f"🚨 ANOMALY DETECTED: {anomaly.get('reason', 'Unknown')}")
                # Log the anomaly
                anomaly_event = event.copy()
                anomaly_event['anomaly'] = anomaly
                with jsonlines.open(self.log_file, mode='a') as writer:
                    writer.write(anomaly_event)
    
    def _create_base_event(self, event_type: str, agent_id: str = "default", **kwargs) -> Dict[str, Any]:
        """Create a base log event with common fields."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "agent_id": agent_id,
            **kwargs
        }
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'content'):
            # Handle LangChain message objects
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
                "additional_kwargs": getattr(obj, 'additional_kwargs', {})
            }
        elif hasattr(obj, '__dict__'):
            # Handle other objects with attributes
            return {
                "type": obj.__class__.__name__,
                "data": {key: self._make_serializable(value) for key, value in obj.__dict__.items()}
            }
        else:
            # Fallback to string representation
            return str(obj)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Log chain start events."""
        if serialized is None:
            serialized = {}
        
        # Convert inputs to JSON-serializable format
        serializable_inputs = self._make_serializable(inputs)
        
        event = self._create_base_event(
            "chain_start",
            agent_id=serialized.get("name", "unknown"),
            inputs=serializable_inputs,
            chain_id=serialized.get("id", "unknown")
        )
        self._write_log(event)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Log chain end events."""
        # Convert outputs to JSON-serializable format
        serializable_outputs = self._make_serializable(outputs)
        
        event = self._create_base_event(
            "chain_end",
            outputs=serializable_outputs
        )
        self._write_log(event)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs) -> None:
        """Log LLM start events."""
        if serialized is None:
            serialized = {}
        
        # Convert prompts to JSON-serializable format
        serializable_prompts = self._make_serializable(prompts)
        
        event = self._create_base_event(
            "llm_start",
            agent_id=serialized.get("name", "unknown"),
            prompts=serializable_prompts,
            llm_id=serialized.get("id", "unknown")
        )
        self._write_log(event)
    
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Log LLM end events."""
        # Convert response to JSON-serializable format
        serializable_response = self._make_serializable(response)
        
        # Extract token usage if available
        token_usage = {}
        if hasattr(response, 'response_metadata'):
            token_usage = response.response_metadata.get('token_usage', {})
        
        event = self._create_base_event(
            "llm_end",
            response=serializable_response,
            token_usage=token_usage
        )
        self._write_log(event)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Log tool start events."""
        if serialized is None:
            serialized = {}
        
        event = self._create_base_event(
            "tool_start",
            agent_id=serialized.get("name", "unknown"),
            tool_input=input_str,
            tool_id=serialized.get("id", "unknown")
        )
        self._write_log(event)
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log tool end events."""
        event = self._create_base_event(
            "tool_end",
            tool_output=output
        )
        self._write_log(event)
    
    def on_agent_action(self, action: Any, **kwargs) -> None:
        """Log agent action events."""
        # Convert action to JSON-serializable format
        serializable_action = self._make_serializable(action)
        
        event = self._create_base_event(
            "agent_action",
            action=serializable_action
        )
        self._write_log(event)
    
    def on_agent_finish(self, finish: Any, **kwargs) -> None:
        """Log agent finish events."""
        # Convert finish to JSON-serializable format
        serializable_finish = self._make_serializable(finish)
        
        event = self._create_base_event(
            "agent_finish",
            finish_reason=serializable_finish
        )
        self._write_log(event)
