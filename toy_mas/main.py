"""
Main entry point for Toy MAS Reliability Framework

Runs a simple agent workflow that generates a Python "Hello World" program
with comprehensive logging and anomaly detection.
"""

import os
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from collector import LogCollector
from detector import AnomalyDetector
from evaluator import PerformanceEvaluator
from optimizer import Optimizer
from tools import tool_registry


class AgentState(TypedDict):
    """State for the agent workflow."""
    messages: List[Any]
    task_completed: bool
    code_generated: str
    verification_result: str


def create_hello_world_agent(config: Dict[str, Any]) -> StateGraph:
    """
    Create a simple agent that generates a Hello World program.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured StateGraph
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model=config["agent"]["model"],
        temperature=config["agent"]["temperature"],
        max_tokens=config["agent"]["max_tokens"],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    def generate_code(state: AgentState) -> AgentState:
        """Generate Python Hello World code."""
        system_prompt = config["prompts"]["hello_world"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please generate a Python Hello World program.")
        ]
        
        response = llm.invoke(messages)
        
        return {
            **state,
            "messages": add_messages(state["messages"], [response]),
            "code_generated": response.content
        }
    
    def verify_code(state: AgentState) -> AgentState:
        """Verify the generated code."""
        code = state.get("code_generated", "")
        
        # Use the code validator tool
        validation_result = tool_registry.run_tool("code_validator", code)
        
        # Simple verification logic
        if "valid" in validation_result.lower() and "hello" in code.lower():
            verification_result = "APPROVED"
        else:
            verification_result = "REJECTED"
        
        return {
            **state,
            "verification_result": verification_result,
            "task_completed": True
        }
    
    def should_continue(state: AgentState) -> str:
        """Determine if the workflow should continue."""
        if state.get("task_completed", False):
            return END
        return "verify_code"
    
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("verify_code", verify_code)
    
    # Add edges
    workflow.add_edge("generate_code", "verify_code")
    workflow.add_conditional_edges(
        "verify_code",
        should_continue,
        {
            END: END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("generate_code")
    
    return workflow


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment results
    """
    # Initialize components
    detector = AnomalyDetector()
    collector = LogCollector(detector=detector)
    
    # Create agent workflow
    agent = create_hello_world_agent(config)
    
    # Compile the workflow
    app = agent.compile()
    
    # Reset detector state
    detector.reset()
    
    # Run the workflow
    initial_state = {
        "messages": [],
        "task_completed": False,
        "code_generated": "",
        "verification_result": ""
    }
    
    try:
        # Run with callbacks
        result = app.invoke(initial_state, config={"callbacks": [collector]})
        
        # Evaluate performance
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate()
        
        return {
            "success": True,
            "result": result,
            "metrics": metrics
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metrics": {"error": f"Experiment failed: {str(e)}"}
        }


def main():
    """Main entry point."""
    print("🚀 Starting Toy MAS Reliability Framework")
    print("="*60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Load default configuration
    try:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "defaults.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    print(f"📋 Configuration loaded:")
    print(f"   Model: {config['agent']['model']}")
    print(f"   Temperature: {config['agent']['temperature']}")
    print(f"   Max Steps: {config['workflow']['max_steps']}")
    
    # Run single experiment
    print(f"\n🧪 Running Hello World experiment...")
    result = run_experiment(config)
    
    if result["success"]:
        print(f"✅ Experiment completed successfully!")
        
        # Print generated code
        generated_code = result["result"].get("code_generated", "")
        if generated_code:
            print(f"\n📝 Generated Code:")
            print("-" * 40)
            print(generated_code)
            print("-" * 40)
        
        # Print verification result
        verification = result["result"].get("verification_result", "")
        print(f"\n🔍 Verification Result: {verification}")
        
        # Print evaluation summary
        evaluator = PerformanceEvaluator()
        evaluator.print_summary(result["metrics"])
        
    else:
        print(f"❌ Experiment failed: {result['error']}")
    
    # Demonstrate optimization (optional)
    print(f"\n🎯 Running optimization demo...")
    optimizer = Optimizer()
    
    # Define search space for optimization
    search_space = {
        "agent.temperature": [0.3, 0.7, 1.0],
        "agent.max_tokens": [500, 1000, 1500]
    }
    
    # Run optimization
    optimization_results = optimizer.optimize(
        run_function=run_experiment,
        search_space=search_space,
        strategy="random",
        max_iterations=3  # Small number for demo
    )
    
    # Print optimization results
    optimizer.print_optimization_results(optimization_results)
    
    print(f"\n🎉 Framework demonstration completed!")
    print(f"📊 Check logs/logs.jsonl for detailed event logs")


if __name__ == "__main__":
    main()
