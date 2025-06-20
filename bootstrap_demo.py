#!/usr/bin/env python3
"""
Demonstration script for bootstrap confidence interval analysis.

This script shows how to use the new bootstrap functionality to analyze
differences between RAG methods with statistical confidence intervals.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from evaluation.bootstrap_analysis import BootstrapAnalyzer
from eda.viz import (
    plot_confidence_intervals, 
    plot_difference_distributions,
    plot_pairwise_heatmap,
    create_comprehensive_bootstrap_report
)


def demo_bootstrap_analysis():
    """
    Demonstrate bootstrap analysis functionality using existing results.
    """
    print("🚀 Bootstrap Confidence Interval Analysis Demo")
    print("=" * 50)
    
    # Find a recent results file
    results_dir = Path("data/results")
    if not results_dir.exists():
        print("❌ No results directory found. Run an evaluation first.")
        return
    
    # Get the most recent results file
    json_files = list(results_dir.glob("evaluation_results_*.json"))
    if not json_files:
        print("❌ No evaluation results found. Run an evaluation first.")
        return
    
    latest_results = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Using results file: {latest_results.name}")
    
    try:
        # Initialize analyzer
        print("\n🔄 Initializing bootstrap analyzer...")
        analyzer = BootstrapAnalyzer(str(latest_results), n_bootstrap=1000)
        
        # Show available methods and metrics
        methods = analyzer.get_available_methods()
        metrics = analyzer.get_available_metrics()
        
        print(f"📊 Available methods: {', '.join(methods)}")
        print(f"📈 Available metrics: {', '.join(metrics)}")
        
        if len(methods) < 2:
            print("⚠️ Need at least 2 methods for comparison. Run evaluation with multiple methods.")
            return
        
        # Basic bootstrap comparison
        print(f"\n🎲 Computing bootstrap confidence intervals...")
        method1, method2 = methods[0], methods[1]
        
        if "rouge1_f" in metrics:
            result = analyzer.bootstrap_difference(method1, method2, "rouge1_f")
            
            print(f"\n📋 Bootstrap Results: {method1} vs {method2} on ROUGE-1")
            print(f"   Observed difference: {result['observed_difference']:.4f}")
            print(f"   95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            print(f"   Statistically significant: {result['is_significant']}")
            print(f"   Effect size (Cohen's d): {result.get('effect_size', 'N/A')}")
        
        # Generate comparison for all method pairs
        print(f"\n📊 Computing all pairwise comparisons...")
        bootstrap_df = analyzer.bootstrap_all_pairs(methods=methods, metrics=metrics[:3])  # Limit to first 3 metrics
        
        print(f"\nSignificant differences found:")
        significant = bootstrap_df[bootstrap_df['is_significant']]
        if not significant.empty:
            for _, row in significant.iterrows():
                print(f"   {row['comparison']} on {row['metric']}: {row['observed_difference']:.4f}")
        else:
            print("   No statistically significant differences detected.")
        
        # Show summary statistics
        print(f"\n📋 Summary Statistics:")
        summary = analyzer.summary_statistics()
        print(summary.head(10).to_string(index=False))
        
        # Check for potential issues
        issues = analyzer.detect_problematic_metrics()
        if issues:
            print(f"\n⚠️ Potential analysis issues detected:")
            for issue_type, problematic in issues.items():
                print(f"   {issue_type}: {len(problematic)} cases")
        
        print(f"\n✅ Basic analysis complete!")
        print(f"\n💡 To create visualizations, use:")
        print(f"   from eda.viz import create_comprehensive_bootstrap_report")
        print(f"   create_comprehensive_bootstrap_report('{latest_results}')")
        
    except ValueError as e:
        if "per_question_metrics" in str(e):
            print("\n❌ Results file doesn't contain per-question metrics.")
            print("   Run a new evaluation with the updated evaluator to generate this data.")
            print("   Example: python3 run_evaluation.py --methods rag ensemble_rerank_rag --num-questions 10")
        else:
            print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def demo_visualization():
    """
    Demonstrate visualization functionality (requires results with per-question metrics).
    """
    results_dir = Path("data/results")
    json_files = list(results_dir.glob("evaluation_results_*.json"))
    
    if not json_files:
        print("❌ No evaluation results found.")
        return
    
    latest_results = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        print(f"\n🎨 Creating comprehensive bootstrap report...")
        create_comprehensive_bootstrap_report(
            str(latest_results),
            output_dir="bootstrap_analysis_output"
        )
    except ValueError as e:
        if "per_question_metrics" in str(e):
            print("❌ Need results with per-question metrics for visualization.")
            print("   Run a new evaluation first.")
        else:
            print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


if __name__ == "__main__":
    # Run basic demo
    demo_bootstrap_analysis()
    
    # Optionally run visualization demo
    print(f"\n" + "="*50)
    response = input("Would you like to create visualizations? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        demo_visualization()
    
    print(f"\n�� Demo complete!") 