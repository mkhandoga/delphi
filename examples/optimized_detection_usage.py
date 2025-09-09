"""
Usage examples and integration guide for the OptimizedDetectionScorer.

This script demonstrates how to use the various optimization features
and provides benchmarking capabilities.
"""
import asyncio
import time
from typing import List, Dict, Any
import logging

from delphi.clients.client import Client
from delphi.latents import LatentRecord
from delphi.scorers.classifier.optimized_detection import (
    OptimizedDetectionScorer, 
    AdaptiveLearningDetector
)
from delphi.scorers.classifier.detection import DetectionScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_latent_record() -> LatentRecord:
    """Create a sample latent record for testing."""
    # This would typically come from your SAE analysis
    return LatentRecord(
        explanation="Words related to American football positions",
        test=[
            "Patriots tight end Rob Gronkowski had his boss",
            "LSU defensive end Isaiah Washington speaks",
            "The quarterback threw a perfect pass",
            "Basketball players dribbled down the court",
            "Soccer midfielder scored the winning goal"
        ],
        not_active=[
            "The weather was sunny today",
            "Programming languages are evolving",
            "Cats like to sleep in the sun"
        ]
    )


async def basic_usage_example(client: Client):
    """Basic usage of the optimized detector."""
    print("\n=== Basic Usage Example ===")
    
    # Initialize with default optimizations
    detector = OptimizedDetectionScorer(
        client=client,
        verbose=True,
        use_dspy=True,
        use_ensemble=True,
        use_dynamic_examples=True,
        use_confidence_weighting=True
    )
    
    # Create test data
    record = create_sample_latent_record()
    
    # Run detection
    start_time = time.time()
    result = await detector(record)
    end_time = time.time()
    
    print(f"Processing time: {end_time - start_time:.2f}s")
    print(f"Total samples processed: {len(result.score)}")
    
    # Analyze results
    correct_predictions = sum(1 for score in result.score if score.correct)
    total_predictions = len([s for s in result.score if s.correct is not None])
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    # Show confidence distribution
    confidences = [s.probability for s in result.score if s.probability is not None]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"Average confidence: {avg_confidence:.3f}")


async def ensemble_comparison_example(client: Client):
    """Compare ensemble vs single approaches."""
    print("\n=== Ensemble Comparison Example ===")
    
    record = create_sample_latent_record()
    
    # Test different configurations
    configs = [
        ("Baseline", {"use_ensemble": False, "use_dspy": False}),
        ("DSPy Only", {"use_ensemble": False, "use_dspy": True}),
        ("Ensemble Only", {"use_ensemble": True, "use_dspy": False}),
        ("Full Optimization", {"use_ensemble": True, "use_dspy": True})
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTesting {name}...")
        
        try:
            detector = OptimizedDetectionScorer(
                client=client,
                verbose=False,
                **config
            )
            
            start_time = time.time()
            result = await detector(record)
            end_time = time.time()
            
            # Calculate metrics
            correct = sum(1 for s in result.score if s.correct)
            total = len([s for s in result.score if s.correct is not None])
            accuracy = correct / total if total > 0 else 0
            
            confidences = [s.probability for s in result.score if s.probability is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            results[name] = {
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "time": end_time - start_time,
                "total_samples": len(result.score)
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Avg Confidence: {avg_confidence:.3f}")
            print(f"  Time: {end_time - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            results[name] = {"error": str(e)}
    
    # Summary comparison
    print("\n=== Summary Comparison ===")
    for name, metrics in results.items():
        if "error" not in metrics:
            print(f"{name:20s} | Acc: {metrics['accuracy']:.3f} | "
                  f"Conf: {metrics['avg_confidence']:.3f} | "
                  f"Time: {metrics['time']:.2f}s")
        else:
            print(f"{name:20s} | ERROR: {metrics['error']}")


async def confidence_weighting_example(client: Client):
    """Demonstrate confidence-aware scoring."""
    print("\n=== Confidence Weighting Example ===")
    
    record = create_sample_latent_record()
    
    # Test with different confidence thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\nTesting confidence threshold: {threshold}")
        
        detector = OptimizedDetectionScorer(
            client=client,
            verbose=False,
            use_confidence_weighting=True,
            confidence_threshold=threshold
        )
        
        result = await detector(record)
        
        # Analyze results
        high_conf_predictions = [
            s for s in result.score 
            if s.probability and s.probability >= threshold
        ]
        
        total_predictions = len(result.score)
        high_conf_count = len(high_conf_predictions)
        
        print(f"  High confidence predictions: {high_conf_count}/{total_predictions}")
        
        if high_conf_predictions:
            high_conf_accuracy = sum(
                1 for s in high_conf_predictions if s.correct
            ) / len(high_conf_predictions)
            print(f"  High confidence accuracy: {high_conf_accuracy:.3f}")


async def active_learning_example(client: Client):
    """Demonstrate active learning capabilities."""
    print("\n=== Active Learning Example ===")
    
    # Use adaptive learning detector
    detector = AdaptiveLearningDetector(
        client=client,
        verbose=False,
        use_ensemble=True,
        confidence_threshold=0.8
    )
    
    record = create_sample_latent_record()
    
    # Run detection
    result = await detector(record)
    
    # Get uncertain examples for human review
    uncertain = detector.get_uncertain_examples(n=5)
    
    print(f"Found {len(uncertain)} uncertain examples:")
    for i, example in enumerate(uncertain):
        print(f"  {i+1}. Text: {example['text'][:50]}...")
        print(f"     Prediction: {example['prediction']}, "
              f"Confidence: {example['confidence']:.3f}")


async def benchmark_vs_original(client: Client):
    """Benchmark against original DetectionScorer."""
    print("\n=== Benchmark vs Original ===")
    
    record = create_sample_latent_record()
    
    # Test original detector
    print("Testing original DetectionScorer...")
    original_detector = DetectionScorer(
        client=client,
        verbose=False,
        n_examples_shown=5
    )
    
    start_time = time.time()
    original_result = await original_detector(record)
    original_time = time.time() - start_time
    
    # Test optimized detector
    print("Testing OptimizedDetectionScorer...")
    optimized_detector = OptimizedDetectionScorer(
        client=client,
        verbose=False,
        n_examples_shown=5,
        use_ensemble=True,
        use_dspy=True
    )
    
    start_time = time.time()
    optimized_result = await optimized_detector(record)
    optimized_time = time.time() - start_time
    
    # Compare results
    def calculate_metrics(result):
        correct = sum(1 for s in result.score if s.correct)
        total = len([s for s in result.score if s.correct is not None])
        accuracy = correct / total if total > 0 else 0
        
        confidences = [s.probability for s in result.score if s.probability is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return accuracy, avg_confidence
    
    orig_acc, orig_conf = calculate_metrics(original_result)
    opt_acc, opt_conf = calculate_metrics(optimized_result)
    
    print(f"\nResults Comparison:")
    print(f"{'Metric':<20} {'Original':<12} {'Optimized':<12} {'Improvement':<12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {orig_acc:<12.3f} {opt_acc:<12.3f} {opt_acc-orig_acc:+.3f}")
    print(f"{'Avg Confidence':<20} {orig_conf:<12.3f} {opt_conf:<12.3f} {opt_conf-orig_conf:+.3f}")
    print(f"{'Time (s)':<20} {original_time:<12.2f} {optimized_time:<12.2f} {optimized_time-original_time:+.2f}")


def dspy_optimization_example():
    """Example of how to use DSPy optimization (when available)."""
    print("\n=== DSPy Optimization Example ===")
    
    try:
        import dspy
        
        # This would be your labeled training data
        # Format: [{"explanation": "...", "examples": "...", "expected": [1,0,1,...]}, ...]
        training_data = [
            {
                "explanation": "American football positions",
                "examples": "Patriots tight end Rob Gronkowski\nBasketball player shoots",
                "expected": [1, 0]
            }
            # Add more training examples...
        ]
        
        print("DSPy training example:")
        print("1. Collect labeled training data")
        print("2. Define accuracy metric")
        print("3. Use BootstrapFewShot teleprompter")
        print("4. Compile optimized model")
        print("\nCode example:")
        print("""
        def accuracy_metric(example, prediction):
            return example.expected == prediction.predictions
        
        teleprompter = dspy.BootstrapFewShot(metric=accuracy_metric)
        optimized = teleprompter.compile(detector.dspy_detector, trainset=training_data)
        """)
        
    except ImportError:
        print("DSPy not available. Install with: pip install dspy")


async def main():
    """Run all examples."""
    # You'll need to implement your actual client
    # This is a placeholder
    class MockClient(Client):
        async def generate(self, prompt, **kwargs):
            # Mock response for demonstration
            from delphi.clients.client import Response
            return Response(text="[1,0,1,0,1]", logprobs=None)
    
    client = MockClient()
    
    try:
        await basic_usage_example(client)
        await ensemble_comparison_example(client)
        await confidence_weighting_example(client)
        await active_learning_example(client)
        await benchmark_vs_original(client)
        dspy_optimization_example()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())