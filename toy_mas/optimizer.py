"""
Optimizer for MAS Reliability Framework

Adjusts workflow parameters to improve reliability using simple search methods.
"""

import json
import random
import os
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from itertools import product
import copy


class Optimizer:
    """
    Optimizes MAS workflow parameters to improve reliability.
    
    Supports different search strategies:
    - Grid search
    - Random search
    - Extensible to Optuna/Ray Tune
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the optimizer.
        
        Args:
            config_file: Path to default configuration file
        """
        if config_file is None:
            # Default to configs directory relative to this file
            config_file = os.path.join(os.path.dirname(__file__), "configs", "defaults.json")
        
        self.config_file = Path(config_file)
        self.default_config = self._load_config()
        self.results = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
            return {
                "agent": {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1000},
                "workflow": {"max_steps": 10, "max_retries": 3},
                "prompts": {"hello_world": "Generate a simple Python 'Hello World' program."}
            }
    
    def optimize(self, 
                 run_function: Callable[[Dict[str, Any]], Dict[str, Any]], 
                 search_space: Dict[str, List[Any]], 
                 strategy: str = "grid",
                 max_iterations: int = 10,
                 objective_metric: str = "success_rate") -> Dict[str, Any]:
        """
        Optimize parameters using specified strategy.
        
        Args:
            run_function: Function that takes config and returns metrics
            search_space: Dictionary of parameter names to lists of values
            strategy: Search strategy ("grid", "random")
            max_iterations: Maximum number of iterations for random search
            objective_metric: Metric to optimize (e.g., "success_rate")
            
        Returns:
            Best configuration and results
        """
        self.results = []
        
        if strategy == "grid":
            configs = self._generate_grid_configs(search_space)
        elif strategy == "random":
            configs = self._generate_random_configs(search_space, max_iterations)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"🔍 Optimizing with {strategy} search over {len(configs)} configurations...")
        
        # Run experiments
        for i, config in enumerate(configs):
            print(f"   Running experiment {i+1}/{len(configs)}...")
            try:
                metrics = run_function(config)
                result = {
                    "config": config,
                    "metrics": metrics,
                    "iteration": i + 1
                }
                self.results.append(result)
                
                # Print progress
                success_rate = metrics.get("performance_metrics", {}).get("success_rate", 0)
                anomaly_rate = metrics.get("anomaly_analysis", {}).get("anomaly_rate", 0)
                print(f"     Success Rate: {success_rate:.2%}, Anomaly Rate: {anomaly_rate:.2%}")
                
            except Exception as e:
                print(f"     ❌ Experiment failed: {e}")
                continue
        
        # Find best configuration
        best_result = self._find_best_result(objective_metric)
        
        return {
            "best_config": best_result["config"] if best_result else None,
            "best_metrics": best_result["metrics"] if best_result else None,
            "all_results": self.results,
            "optimization_summary": self._create_summary()
        }
    
    def _generate_grid_configs(self, search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        # Get all parameter combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        configs = []
        for combination in product(*param_values):
            config = copy.deepcopy(self.default_config)
            
            # Update config with parameter values
            for param_name, param_value in zip(param_names, combination):
                self._set_nested_config(config, param_name, param_value)
            
            configs.append(config)
        
        return configs
    
    def _generate_random_configs(self, search_space: Dict[str, List[Any]], max_iterations: int) -> List[Dict[str, Any]]:
        """Generate random configurations for random search."""
        configs = []
        
        for _ in range(max_iterations):
            config = copy.deepcopy(self.default_config)
            
            # Randomly select values for each parameter
            for param_name, param_values in search_space.items():
                param_value = random.choice(param_values)
                self._set_nested_config(config, param_name, param_value)
            
            configs.append(config)
        
        return configs
    
    def _set_nested_config(self, config: Dict[str, Any], param_name: str, param_value: Any) -> None:
        """Set nested configuration parameter (e.g., "agent.temperature")."""
        if "." in param_name:
            # Handle nested parameters like "agent.temperature"
            parts = param_name.split(".")
            current = config
            
            # Navigate to the parent dictionary
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the final value
            current[parts[-1]] = param_value
        else:
            # Handle top-level parameters
            config[param_name] = param_value
    
    def _find_best_result(self, objective_metric: str) -> Optional[Dict[str, Any]]:
        """Find the best result based on objective metric."""
        if not self.results:
            return None
        
        best_result = None
        best_score = float('-inf')
        
        for result in self.results:
            metrics = result.get("metrics", {})
            
            # Navigate nested metrics to find the objective
            score = self._extract_metric_value(metrics, objective_metric)
            
            if score is not None and score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_path: str) -> Optional[float]:
        """Extract metric value from nested dictionary."""
        try:
            # Handle nested paths like "performance_metrics.success_rate"
            parts = metric_path.split(".")
            current = metrics
            
            for part in parts:
                current = current[part]
            
            return float(current) if current is not None else None
        except (KeyError, TypeError, ValueError):
            return None
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create optimization summary."""
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Extract metrics for analysis
        success_rates = []
        anomaly_rates = []
        
        for result in self.results:
            metrics = result.get("metrics", {})
            success_rate = self._extract_metric_value(metrics, "performance_metrics.success_rate")
            anomaly_rate = self._extract_metric_value(metrics, "anomaly_analysis.anomaly_rate")
            
            if success_rate is not None:
                success_rates.append(success_rate)
            if anomaly_rate is not None:
                anomaly_rates.append(anomaly_rate)
        
        return {
            "total_experiments": len(self.results),
            "successful_experiments": len(success_rates),
            "average_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "average_anomaly_rate": sum(anomaly_rates) / len(anomaly_rates) if anomaly_rates else 0,
            "best_success_rate": max(success_rates) if success_rates else 0,
            "worst_success_rate": min(success_rates) if success_rates else 0
        }
    
    def print_optimization_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Print formatted optimization results."""
        if results is None:
            if not self.results:
                print("❌ No optimization results available")
                return
            results = {
                "best_config": self._find_best_result("performance_metrics.success_rate"),
                "optimization_summary": self._create_summary()
            }
        
        print("\n" + "="*60)
        print("🎯 OPTIMIZATION RESULTS")
        print("="*60)
        
        # Summary
        summary = results.get("optimization_summary", {})
        if "error" not in summary:
            print(f"\n📊 Summary:")
            print(f"   Total Experiments: {summary.get('total_experiments', 0)}")
            print(f"   Successful Experiments: {summary.get('successful_experiments', 0)}")
            print(f"   Average Success Rate: {summary.get('average_success_rate', 0):.2%}")
            print(f"   Best Success Rate: {summary.get('best_success_rate', 0):.2%}")
            print(f"   Average Anomaly Rate: {summary.get('average_anomaly_rate', 0):.2%}")
        
        # Best configuration
        best_result = results.get("best_config")
        if best_result:
            print(f"\n🏆 Best Configuration:")
            best_config = best_result.get("config", {})
            best_metrics = best_result.get("metrics", {})
            
            # Print key configuration parameters
            for section, params in best_config.items():
                if isinstance(params, dict):
                    print(f"   {section}:")
                    for key, value in params.items():
                        print(f"     {key}: {value}")
            
            # Print best metrics
            success_rate = self._extract_metric_value(best_metrics, "performance_metrics.success_rate")
            anomaly_rate = self._extract_metric_value(best_metrics, "anomaly_analysis.anomaly_rate")
            
            print(f"\n   Best Performance:")
            print(f"     Success Rate: {success_rate:.2%}" if success_rate is not None else "     Success Rate: N/A")
            print(f"     Anomaly Rate: {anomaly_rate:.2%}" if anomaly_rate is not None else "     Anomaly Rate: N/A")
        else:
            print(f"\n❌ No successful experiments found")
        
        print("\n" + "="*60)


class OptunaOptimizer(Optimizer):
    """
    Optuna-based optimizer (placeholder for future implementation).
    
    This would integrate with Optuna for advanced optimization.
    """
    
    def __init__(self, config_file: str = "configs/defaults.json"):
        """Initialize Optuna optimizer."""
        super().__init__(config_file)
        # Future: Initialize Optuna study
    
    def optimize(self, 
                 run_function: Callable[[Dict[str, Any]], Dict[str, Any]], 
                 search_space: Dict[str, List[Any]], 
                 strategy: str = "optuna",
                 max_iterations: int = 100,
                 objective_metric: str = "success_rate") -> Dict[str, Any]:
        """
        Optimize using Optuna (placeholder implementation).
        
        For now, falls back to random search.
        """
        print("⚠️  Optuna optimizer not implemented yet, falling back to random search")
        return super().optimize(run_function, search_space, "random", max_iterations, objective_metric)
