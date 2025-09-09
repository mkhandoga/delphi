#!/usr/bin/env python3
"""
Prompt Experiment Workflow Helper

Streamlined workflow for testing prompt optimizations.
"""

import argparse
from pathlib import Path
from experiment_tracker import run_experiment_pipeline, record_results, ExperimentTracker
from delphi.explainers.default.prompts import SYSTEM

def record_current_prompt_experiment(version_name: str, description: str):
    """Record the current prompt as a baseline experiment."""
    
    # Get current prompt from prompts.py
    current_prompt = SYSTEM
    
    experiment_id = run_experiment_pipeline(
        prompt_text=current_prompt,
        prompt_version=version_name,
        description=description,
        run_name="current_system",
        model="unknown"
    )
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Run Delphi with current prompt:")
    print(f"   python -m delphi --graph gemma-michael-jordan_2025-09-09T16-08-55-746Z.json --name {version_name}_run")
    print(f"2. Evaluate and record results:")
    print(f"   python label_evaluation.py --explanations-dir results/{version_name}_run/explanations --output {version_name}_results.csv --experiment-id {experiment_id}")
    
    return experiment_id

def show_experiment_status():
    """Show current experiment status."""
    
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments()
    
    print("🧪 EXPERIMENT STATUS")
    print("=" * 50)
    
    if not experiments:
        print("No experiments recorded yet.")
        return
    
    print(f"Total experiments: {len(experiments)}")
    print("\nRecent experiments:")
    
    for exp in experiments[:10]:  # Show last 10
        exp_data = tracker.get_experiment(exp['experiment_id'])
        if exp_data:
            results = exp_data.get('results', {})
            has_results = bool(results)
            
            status = "✅ Complete" if has_results else "⏳ Pending"
            
            print(f"\n{status} {exp['prompt_version']}")
            print(f"  ID: {exp['experiment_id']}")
            print(f"  Description: {exp['description']}")
            
            if has_results:
                print(f"  📊 Semantic sim: {results.get('avg_semantic_similarity', 0):.3f}")
                print(f"  📏 Word ratio: {results.get('avg_word_count_ratio', 0):.1f}x")
                print(f"  🎯 Length score: {results.get('avg_length_score', 0):.3f}")

def compare_top_experiments(top_n: int = 5):
    """Compare the top N experiments by semantic similarity."""
    
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments()
    
    # Get experiments with results
    exp_with_results = []
    for exp_info in experiments:
        exp_data = tracker.get_experiment(exp_info['experiment_id'])
        if exp_data and exp_data.get('results'):
            results = exp_data['results']
            exp_with_results.append({
                'experiment_id': exp_data['experiment_id'],
                'version': exp_data['prompt_version'],
                'description': exp_data['description'],
                'semantic_sim': results.get('avg_semantic_similarity', 0),
                'word_ratio': results.get('avg_word_count_ratio', 0),
                'length_score': results.get('avg_length_score', 0),
                'timestamp': exp_data['timestamp']
            })
    
    if not exp_with_results:
        print("❌ No experiments with results found")
        return
    
    # Sort by semantic similarity
    exp_with_results.sort(key=lambda x: x['semantic_sim'], reverse=True)
    
    print(f"🏆 TOP {min(top_n, len(exp_with_results))} EXPERIMENTS (by semantic similarity)")
    print("=" * 80)
    
    for i, exp in enumerate(exp_with_results[:top_n]):
        print(f"\n#{i+1} {exp['version']} (sim: {exp['semantic_sim']:.3f})")
        print(f"    ID: {exp['experiment_id']}")
        print(f"    Description: {exp['description'][:60]}{'...' if len(exp['description']) > 60 else ''}")
        print(f"    Metrics: Word ratio {exp['word_ratio']:.1f}x, Length score {exp['length_score']:.3f}")
        print(f"    Date: {exp['timestamp'][:10]}")

def quick_baseline_setup():
    """Quick setup to record current system as baseline."""
    
    print("🚀 QUICK BASELINE SETUP")
    print("=" * 30)
    
    # Record current prompt
    baseline_id = record_current_prompt_experiment(
        version_name="baseline_optimized",
        description="Optimized prompt currently deployed in prompts.py (3-word max, concept focus)"
    )
    
    print(f"\n✅ Baseline recorded as: {baseline_id}")
    print(f"\n🔄 Run the suggested commands to complete the baseline measurement.")

def main():
    parser = argparse.ArgumentParser(description="Prompt experiment workflow helper")
    parser.add_argument("--status", action="store_true", help="Show experiment status")
    parser.add_argument("--compare", type=int, default=5, help="Compare top N experiments")
    parser.add_argument("--baseline", action="store_true", help="Quick baseline setup")
    parser.add_argument("--record-current", nargs=2, metavar=('VERSION', 'DESCRIPTION'), 
                       help="Record current prompt as experiment")
    
    args = parser.parse_args()
    
    if args.status:
        show_experiment_status()
    elif args.compare:
        compare_top_experiments(args.compare)
    elif args.baseline:
        quick_baseline_setup()
    elif args.record_current:
        version, description = args.record_current
        record_current_prompt_experiment(version, description)
    else:
        print("🧪 PROMPT EXPERIMENT WORKFLOW")
        print("=" * 40)
        print("Available commands:")
        print("  --status          Show current experiments")
        print("  --compare N       Compare top N experiments")  
        print("  --baseline        Quick baseline setup")
        print("  --record-current VERSION DESCRIPTION    Record current prompt")
        print("")
        print("Example workflow:")
        print("1. python prompt_experiment_workflow.py --baseline")
        print("2. [Run suggested commands]")
        print("3. [Modify prompt in prompts.py]")
        print("4. python prompt_experiment_workflow.py --record-current v2_stricter 'Even stricter 1-word labels'")
        print("5. [Run suggested commands]")
        print("6. python prompt_experiment_workflow.py --compare 5")

if __name__ == "__main__":
    main()