#!/usr/bin/env python3
"""
Bootstrap confidence interval analysis for RAG evaluation metrics.

This module provides functionality to compute bootstrap confidence intervals
on metric differences between different RAG methods, enabling statistical
comparison of their performance.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import warnings


class BootstrapAnalyzer:
    """
    Handles bootstrap confidence interval computation for metric differences
    between RAG evaluation methods.
    """
    
    def __init__(self, results_path: str, n_bootstrap: int = 1000, random_seed: int = 42):
        """
        Initialize the bootstrap analyzer.
        
        Args:
            results_path: Path to the evaluation results JSON file
            n_bootstrap: Number of bootstrap samples to generate
            random_seed: Random seed for reproducibility
        """
        self.results_path = Path(results_path)
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.results = None
        self.per_question_df = None
        
        np.random.seed(random_seed)
        
        # Load results immediately
        self.load_results()
    
    def load_results(self) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.
        
        Returns:
            Dictionary containing evaluation results
        """
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        if "per_question_metrics" not in self.results:
            raise ValueError(
                "Results file does not contain per-question metrics. "
                "Please run evaluation with updated evaluator to generate per-question data."
            )
        
        return self.results
    
    def extract_per_question_metrics(self) -> pd.DataFrame:
        """
        Extract per-question metrics into a structured DataFrame.
        
        Returns:
            DataFrame with columns: [question_id, method, metric, value]
        """
        if self.per_question_df is not None:
            return self.per_question_df
        
        rows = []
        per_question_data = self.results["per_question_metrics"]
        
        for method, metrics in per_question_data.items():
            for metric, values in metrics.items():
                for question_id, value in enumerate(values):
                    rows.append({
                        'question_id': question_id,
                        'method': method,
                        'metric': metric,
                        'value': value
                    })
        
        self.per_question_df = pd.DataFrame(rows)
        return self.per_question_df
    
    def get_available_methods(self) -> List[str]:
        """Get list of available evaluation methods."""
        return list(self.results["per_question_metrics"].keys())
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        if not self.results["per_question_metrics"]:
            return []
        
        # Get metrics from the first method
        first_method = next(iter(self.results["per_question_metrics"].values()))
        return list(first_method.keys())
    
    def bootstrap_difference(
        self, 
        method1: str, 
        method2: str, 
        metric: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compute bootstrap confidence interval for the difference in performance
        between two methods on a specific metric.
        
        Args:
            method1: Name of the first method
            method2: Name of the second method  
            metric: Name of the metric to compare
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Dictionary containing bootstrap results
        """
        # Validate inputs
        available_methods = self.get_available_methods()
        if method1 not in available_methods:
            raise ValueError(f"Method '{method1}' not found. Available: {available_methods}")
        if method2 not in available_methods:
            raise ValueError(f"Method '{method2}' not found. Available: {available_methods}")
        
        available_metrics = self.get_available_metrics()
        if metric not in available_metrics:
            raise ValueError(f"Metric '{metric}' not found. Available: {available_metrics}")
        
        # Get metric values for both methods
        values1 = np.array(self.results["per_question_metrics"][method1][metric])
        values2 = np.array(self.results["per_question_metrics"][method2][metric])
        
        if len(values1) != len(values2):
            raise ValueError(f"Mismatched number of questions: {len(values1)} vs {len(values2)}")
        
        n_questions = len(values1)
        
        # Compute observed difference
        observed_diff = np.mean(values1) - np.mean(values2)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_questions, size=n_questions, replace=True)
            
            boot_values1 = values1[indices]
            boot_values2 = values2[indices]
            
            boot_diff = np.mean(boot_values1) - np.mean(boot_values2)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute confidence interval
        ci_lower, ci_upper = self.get_confidence_interval(bootstrap_diffs, alpha)
        
        # Determine statistical significance
        is_significant = ci_lower > 0 or ci_upper < 0
        
        return {
            'method1': method1,
            'method2': method2,
            'metric': metric,
            'observed_difference': observed_diff,
            'bootstrap_differences': bootstrap_diffs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': 1 - alpha,
            'is_significant': is_significant,
            'n_bootstrap': self.n_bootstrap,
            'n_questions': n_questions,
            'method1_mean': np.mean(values1),
            'method2_mean': np.mean(values2),
            'method1_std': np.std(values1),
            'method2_std': np.std(values2)
        }
    
    def bootstrap_all_pairs(
        self, 
        methods: Optional[List[str]] = None, 
        metrics: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Compute bootstrap confidence intervals for all pairs of methods
        across multiple metrics.
        
        Args:
            methods: List of methods to compare (default: all available)
            metrics: List of metrics to analyze (default: all available)
            alpha: Significance level
            
        Returns:
            DataFrame with bootstrap results for all method pairs
        """
        if methods is None:
            methods = self.get_available_methods()
        if metrics is None:
            metrics = self.get_available_metrics()
        
        results = []
        
        # Compare all pairs of methods
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                
                for metric in metrics:
                    try:
                        bootstrap_result = self.bootstrap_difference(
                            method1, method2, metric, alpha
                        )
                        
                        results.append({
                            'method1': method1,
                            'method2': method2,
                            'metric': metric,
                            'observed_difference': bootstrap_result['observed_difference'],
                            'ci_lower': bootstrap_result['ci_lower'],
                            'ci_upper': bootstrap_result['ci_upper'],
                            'confidence_level': bootstrap_result['confidence_level'],
                            'is_significant': bootstrap_result['is_significant'],
                            'method1_mean': bootstrap_result['method1_mean'],
                            'method2_mean': bootstrap_result['method2_mean'],
                            'effect_size': self._calculate_effect_size(bootstrap_result),
                            'comparison': f"{method1} vs {method2}"
                        })
                    except Exception as e:
                        warnings.warn(f"Failed to compute bootstrap for {method1} vs {method2} on {metric}: {e}")
        
        return pd.DataFrame(results)
    
    def get_confidence_interval(
        self, 
        bootstrap_samples: np.ndarray, 
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Compute confidence interval from bootstrap samples.
        
        Args:
            bootstrap_samples: Array of bootstrap statistics
            alpha: Significance level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        percentile_lower = (alpha / 2) * 100
        percentile_upper = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_samples, percentile_lower)
        ci_upper = np.percentile(bootstrap_samples, percentile_upper)
        
        return ci_lower, ci_upper
    
    def _calculate_effect_size(self, bootstrap_result: Dict[str, Any]) -> float:
        """
        Calculate Cohen's d effect size for the difference.
        
        Args:
            bootstrap_result: Results from bootstrap_difference()
            
        Returns:
            Effect size (Cohen's d)
        """
        mean_diff = bootstrap_result['observed_difference']
        
        # Pooled standard deviation
        var1 = bootstrap_result['method1_std'] ** 2
        var2 = bootstrap_result['method2_std'] ** 2
        pooled_std = np.sqrt((var1 + var2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for all methods and metrics.
        
        Returns:
            DataFrame with mean, std, min, max for each method-metric combination
        """
        df = self.extract_per_question_metrics()
        
        summary = df.groupby(['method', 'metric'])['value'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        return summary.reset_index()
    
    def detect_problematic_metrics(self) -> Dict[str, List[str]]:
        """
        Detect metrics that might be problematic for bootstrap analysis.
        
        Returns:
            Dictionary mapping issue types to lists of problematic metrics
        """
        issues = defaultdict(list)
        
        for metric in self.get_available_metrics():
            for method in self.get_available_methods():
                values = np.array(self.results["per_question_metrics"][method][metric])
                
                # Check for constant values
                if np.std(values) == 0:
                    issues['constant_values'].append(f"{method}:{metric}")
                
                # Check for extreme outliers (beyond 3 standard deviations)
                z_scores = np.abs((values - np.mean(values)) / (np.std(values) + 1e-10))
                if np.any(z_scores > 3):
                    issues['extreme_outliers'].append(f"{method}:{metric}")
                
                # Check for binary metrics (only 0 and 1)
                unique_values = np.unique(values)
                if len(unique_values) == 2 and set(unique_values) == {0.0, 1.0}:
                    issues['binary_metrics'].append(f"{method}:{metric}")
        
        return dict(issues) 