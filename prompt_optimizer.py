#!/usr/bin/env python3
"""
Prompt Optimization for Delphi Label Generation

Creates improved prompts based on evaluation insights and golden label examples.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import orjson
import random
from label_evaluation import extract_golden_features

def extract_golden_examples(golden_features: Dict, n_examples: int = 10) -> List[Tuple[str, str]]:
    """Extract high-quality examples from golden labels for few-shot learning."""
    
    # Categories of good examples to include
    examples = []
    
    # Short, specific concepts (ideal target)
    short_specific = [(k, v) for k, v in golden_features.items() if len(v.split()) <= 2 and not any(generic in v.lower() for generic in ['various', 'common', 'often'])]
    if short_specific:
        examples.extend(random.sample(short_specific[:20], min(3, len(short_specific))))
    
    # Technical terms
    technical = [(k, v) for k, v in golden_features.items() if any(tech in v.lower() for tech in ['code', 'syntax', 'html', 'json', 'api', 'http'])]
    if technical:
        examples.extend(random.sample(technical[:10], min(2, len(technical))))
    
    # Names and proper nouns
    names = [(k, v) for k, v in golden_features.items() if any(name in v.lower() for name in ['name', 'proper', 'person'])]
    if names:
        examples.extend(random.sample(names[:10], min(2, len(names))))
    
    # Punctuation and formatting
    punct = [(k, v) for k, v in golden_features.items() if any(p in v.lower() for p in ['punctuation', 'colon', 'period', 'bracket'])]
    if punct:
        examples.extend(random.sample(punct[:10], min(2, len(punct))))
    
    # General categories (but still specific)
    general = [(k, v) for k, v in golden_features.items() if len(v.split()) <= 3]
    if general and len(examples) < n_examples:
        remaining = n_examples - len(examples)
        examples.extend(random.sample(general[:20], min(remaining, len(general))))
    
    return examples[:n_examples]

def create_baseline_prompt() -> str:
    """Create the current baseline prompt (extracted from the explainer)."""
    
    # This would be the current explainer prompt - placeholder for now
    return """Based on the following examples where this feature activates, provide a detailed explanation of what pattern or concept this feature detects.

Examples:
{examples}

Explanation:"""

def create_optimized_prompt_v1(golden_examples: List[Tuple]) -> str:
    """Create first optimized prompt focusing on conciseness."""
    
    examples_text = "\n".join([f"Feature: {k} → Label: \"{v}\"" for k, v in golden_examples])
    
    return f"""You are a precise feature labeling expert. Your task is to create a concise, specific label for a neural network feature.

GOOD LABELS (examples):
{examples_text}

RULES:
- Use 1-3 words maximum
- Be specific, not generic (avoid "various", "common", "often")
- Focus on the core concept, not descriptions
- Use technical terms when appropriate

BAD examples:
- "Various words and tokens that often appear in text" → TOO GENERIC
- "Common patterns frequently seen in different contexts" → TOO VAGUE
- "Words that are contextually significant and meaningful" → MEANINGLESS

Given these activation examples, provide a precise 1-3 word label:

Examples:
{{examples}}

Label (1-3 words):"""

def create_optimized_prompt_v2(golden_examples: List[Tuple]) -> str:
    """Create second optimized prompt with chain-of-thought."""
    
    examples_text = "\n".join([f"Examples activate on: [pattern] → Label: \"{v}\"" for k, v in golden_examples[:5]])
    
    return f"""You are a neural network interpretability expert. Label this feature with precision.

HIGH-QUALITY LABELS:
{examples_text}

PROCESS:
1. First, identify the specific pattern in the examples
2. Then, create a 1-3 word label for this pattern

CONSTRAINTS:
- Maximum 3 words
- Avoid generic terms (various, common, often, specific, different)
- Focus on the precise concept, not meta-descriptions

Examples where this feature activates:
{{examples}}

Step 1 - Pattern I see: [describe the specific pattern in 1 sentence]
Step 2 - Precise label: [1-3 words]"""

def create_optimized_prompt_v3(golden_examples: List[Tuple]) -> str:
    """Create third optimized prompt with contrastive examples."""
    
    good_examples = golden_examples[:4]
    
    return f"""Label this neural network feature with a concise, precise term.

EXCELLENT LABELS (study these):
{chr(10).join([f'• "{v}"' for k, v in good_examples])}

TERRIBLE LABELS (never do this):
• "Words and phrases that are contextually significant"
• "Various tokens that often appear in different settings"  
• "Common patterns frequently used in text"

RULES:
✅ 1-3 words maximum
✅ Specific concept (e.g., "sports scores", "HTML tags")
✅ Technical precision when applicable
❌ No generic descriptors ("various", "common", "often")
❌ No meta-commentary about the text

Feature activation examples:
{{examples}}

Concise label:"""

def create_optimized_prompt_v4(golden_examples: List[Tuple]) -> str:
    """Create fourth optimized prompt with scoring criteria."""
    
    return f"""Create a precise label for this neural network feature. 

SCORE YOUR LABEL (aim for 5/5):
5/5: Specific concept in 1-2 words (e.g., "basketball scores", "Python imports")
4/5: Good concept, slightly verbose (e.g., "sports statistics") 
3/5: Related concept but generic (e.g., "numerical data")
2/5: Vague description (e.g., "contextually relevant numbers")
1/5: Meaningless (e.g., "various tokens that often appear")

Your label must score 4/5 or higher.

Feature examples:
{{examples}}

Label (aim for 5/5):"""

def test_prompts_on_examples(golden_features: Dict, sample_size: int = 5):
    """Test different prompts on sample features to preview improvements."""
    
    print("=== PROMPT TESTING PREVIEW ===\n")
    
    # Get examples for few-shot learning
    golden_examples = extract_golden_examples(golden_features, 8)
    
    # Sample features to test prompts on
    test_features = random.sample(list(golden_features.items()), min(sample_size, len(golden_features)))
    
    prompts = {
        "Baseline": create_baseline_prompt(),
        "Optimized V1 (Conciseness)": create_optimized_prompt_v1(golden_examples),
        "Optimized V2 (Chain-of-thought)": create_optimized_prompt_v2(golden_examples), 
        "Optimized V3 (Contrastive)": create_optimized_prompt_v3(golden_examples),
        "Optimized V4 (Self-scoring)": create_optimized_prompt_v4(golden_examples)
    }
    
    print("GOLDEN EXAMPLES BEING USED:")
    for k, v in golden_examples:
        print(f"  {k}: \"{v}\"")
    
    print(f"\nPROMPT VARIATIONS (showing how they would format):\n")
    
    example_activations = "Example 1: 'basketball game tonight'\nExample 2: 'final score 102-98'\nExample 3: 'Lakers win'"
    
    for name, prompt_template in prompts.items():
        print(f"--- {name} ---")
        formatted_prompt = prompt_template.replace("{examples}", example_activations)
        print(formatted_prompt[:300] + "..." if len(formatted_prompt) > 300 else formatted_prompt)
        print()

def save_optimized_prompts(output_dir: Path):
    """Save all optimized prompt templates to files."""
    
    output_dir.mkdir(exist_ok=True)
    
    # This would be integrated into the actual explainer configuration
    prompt_configs = {
        "concise_prompt_v1": {
            "name": "Conciseness Focus",
            "template": create_optimized_prompt_v1([]),  # Would be filled with real examples
            "description": "Emphasizes 1-3 word labels and avoids generic language"
        },
        "chain_of_thought_v2": {
            "name": "Chain-of-thought Reasoning", 
            "template": create_optimized_prompt_v2([]),
            "description": "Two-step process: identify pattern, then label"
        },
        "contrastive_v3": {
            "name": "Contrastive Examples",
            "template": create_optimized_prompt_v3([]),
            "description": "Shows good vs bad examples explicitly"
        },
        "self_scoring_v4": {
            "name": "Self-scoring Criteria",
            "template": create_optimized_prompt_v4([]),
            "description": "Asks model to score its own output for quality"
        }
    }
    
    for name, config in prompt_configs.items():
        with open(output_dir / f"{name}.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Saved {len(prompt_configs)} optimized prompt templates to {output_dir}/")

def create_ablation_study_config() -> Dict:
    """Create configuration for systematic prompt A/B testing."""
    
    return {
        "experiment_name": "label_quality_improvement",
        "baseline_prompt": "current_explainer_prompt",
        "test_conditions": [
            {
                "name": "conciseness_constraints", 
                "prompt": "optimized_v1",
                "expected_improvements": ["reduced_verbosity", "shorter_labels"]
            },
            {
                "name": "chain_of_thought",
                "prompt": "optimized_v2", 
                "expected_improvements": ["better_reasoning", "more_accurate_concepts"]
            },
            {
                "name": "contrastive_learning",
                "prompt": "optimized_v3",
                "expected_improvements": ["fewer_generic_terms", "clearer_distinctions"]
            },
            {
                "name": "self_evaluation",
                "prompt": "optimized_v4",
                "expected_improvements": ["higher_self_awareness", "quality_control"]
            }
        ],
        "evaluation_metrics": [
            "word_count_ratio",
            "semantic_similarity", 
            "generic_term_frequency",
            "llm_judge_rating"
        ],
        "sample_size_per_condition": 50,
        "significance_threshold": 0.05
    }

def main():
    parser = argparse.ArgumentParser(description="Generate optimized prompts for label generation")
    parser.add_argument(
        "--golden-graph",
        type=Path,
        default=Path("gemma-michael-jordan_2025-09-09T16-08-55-746Z.json"),
        help="Path to golden labels graph"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimized_prompts"),
        help="Directory to save optimized prompts"
    )
    parser.add_argument(
        "--test-preview",
        action="store_true",
        help="Show preview of how different prompts would format"
    )
    
    args = parser.parse_args()
    
    if not args.golden_graph.exists():
        print(f"Golden graph file not found: {args.golden_graph}")
        return
    
    # Load golden features
    golden_features = extract_golden_features(args.golden_graph)
    print(f"Loaded {len(golden_features)} golden features")
    
    if args.test_preview:
        test_prompts_on_examples(golden_features)
    
    # Save optimized prompts
    save_optimized_prompts(args.output_dir)
    
    # Create A/B testing configuration
    ablation_config = create_ablation_study_config()
    with open(args.output_dir / "ablation_study_config.json", 'w') as f:
        json.dump(ablation_config, f, indent=2)
    
    print(f"\n✅ Generated optimized prompts in {args.output_dir}/")
    print("📋 Next steps:")
    print("1. Review the generated prompts")
    print("2. Integrate one into your explainer configuration")  
    print("3. Run A/B test using the ablation study config")
    print("4. Measure improvements using the evaluation script")

if __name__ == "__main__":
    main()