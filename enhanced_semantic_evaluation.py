#!/usr/bin/env python3
"""
Enhanced Semantic Similarity Evaluation

Replaces basic word overlap with proper semantic similarity measures.
"""

import numpy as np
from typing import Dict, List, Tuple
import re
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class SemanticEvaluator:
    """Enhanced semantic similarity evaluation for label quality."""
    
    def __init__(self, method='embedding'):
        self.method = method
        self.model = None
        
        if method == 'embedding' and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use a model good for short phrases/concepts on GPUs 6 or 7
            import torch
            if torch.cuda.is_available():
                device = 'cuda:6' if torch.cuda.device_count() > 6 else 'cuda:7' if torch.cuda.device_count() > 7 else 'cuda:0'
            else:
                device = 'cpu'
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
    def calculate_embedding_similarity(self, golden: str, autogen: str) -> float:
        """Use sentence embeddings for semantic similarity."""
        if not self.model:
            return self.fallback_similarity(golden, autogen)
            
        # Clean the labels
        golden_clean = self.clean_label(golden)
        autogen_clean = self.clean_label(autogen)
        
        # Get embeddings
        embeddings = self.model.encode([golden_clean, autogen_clean])
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def calculate_llm_similarity(self, golden: str, autogen: str) -> float:
        """Use LLM to judge semantic similarity."""
        prompt = f"""Rate the semantic similarity between these two feature labels on a scale of 0.0 to 1.0:

Golden label: "{golden}"
Generated label: "{autogen}"

Consider:
- Do they refer to the same concept?
- Are they semantically equivalent even if worded differently?
- Would an expert consider them the same thing?

Examples:
- "names" vs "proper nouns" = 0.9 (same concept, different terms)
- "colon punctuation" vs "colon symbols" = 0.95 (nearly identical)
- "sports scores" vs "numerical data" = 0.3 (related but not same concept)
- "facts" vs "various contextual tokens" = 0.1 (completely different)

Return only a number between 0.0 and 1.0:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            # Extract number from response
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'[0-1]?\.\d+', score_text)[0])
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            print(f"LLM similarity failed: {e}")
            return self.fallback_similarity(golden, autogen)
    
    def calculate_concept_similarity(self, golden: str, autogen: str) -> float:
        """Rule-based semantic similarity for common interpretability concepts."""
        
        golden_clean = self.clean_label(golden).lower()
        autogen_clean = self.clean_label(autogen).lower()
        
        # Concept equivalence mappings
        concept_groups = {
            'names': ['names', 'proper nouns', 'proper names', 'person names', 'individuals'],
            'punctuation': ['punctuation', 'colon', 'period', 'comma', 'bracket', 'symbol'],
            'code': ['code', 'syntax', 'programming', 'html', 'css', 'javascript', 'python'],
            'numbers': ['numbers', 'digits', 'numerical', 'numeric', 'scores', 'counts'],
            'verbs': ['verbs', 'actions', 'doing words', 'action words'],
            'adjectives': ['adjectives', 'describing words', 'descriptive'],
            'prepositions': ['prepositions', 'position words', 'directional words'],
        }
        
        # Check if both labels belong to same concept group
        golden_group = None
        autogen_group = None
        
        for group, terms in concept_groups.items():
            if any(term in golden_clean for term in terms):
                golden_group = group
            if any(term in autogen_clean for term in terms):
                autogen_group = group
        
        if golden_group and autogen_group:
            if golden_group == autogen_group:
                return 0.8  # Same concept family
            else:
                return 0.1  # Different concept families
        
        # Check for exact word matches in different forms
        golden_words = set(golden_clean.split())
        autogen_words = set(autogen_clean.split())
        
        if golden_words.intersection(autogen_words):
            return 0.4  # Some word overlap
        
        # Check for single word golden labels
        if len(golden_words) == 1:
            golden_word = list(golden_words)[0]
            if golden_word in autogen_clean:
                return 0.6  # Golden concept mentioned in autogen
        
        return 0.0  # No similarity found
    
    def fallback_similarity(self, golden: str, autogen: str) -> float:
        """Fallback to word overlap similarity."""
        golden_words = set(golden.lower().split())
        autogen_words = set(autogen.lower().split())
        
        if not golden_words and not autogen_words:
            return 1.0
        
        intersection = golden_words.intersection(autogen_words)
        union = golden_words.union(autogen_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def clean_label(self, label: str) -> str:
        """Clean label text for comparison."""
        # Remove quotes and extra whitespace
        label = re.sub(r'^["\'\s]+|["\'\s]+$', '', label)
        # Remove common filler words for semantic comparison
        filler_words = ['the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'that', 'this']
        words = label.split()
        cleaned_words = [w for w in words if w.lower() not in filler_words]
        return ' '.join(cleaned_words) if cleaned_words else label
    
    def evaluate(self, golden: str, autogen: str) -> Dict[str, float]:
        """Run multiple similarity evaluations and return combined results."""
        
        results = {
            'word_overlap': self.fallback_similarity(golden, autogen),
            'concept_based': self.calculate_concept_similarity(golden, autogen),
        }
        
        if self.method == 'embedding' and SENTENCE_TRANSFORMERS_AVAILABLE:
            results['embedding'] = self.calculate_embedding_similarity(golden, autogen)
            results['combined'] = (
                0.3 * results['word_overlap'] + 
                0.4 * results['concept_based'] + 
                0.3 * results['embedding']
            )
        else:
            results['combined'] = (
                0.4 * results['word_overlap'] + 
                0.6 * results['concept_based']
            )
        
        return results

def test_semantic_evaluator():
    """Test the semantic evaluator on known examples."""
    
    evaluator = SemanticEvaluator()
    
    test_cases = [
        ("names", "proper nouns and entities", "Should be high similarity"),
        ("colon punctuation", "punctuation marks that introduce lists", "Should be high similarity"),
        ("facts", "various contextual tokens", "Should be low similarity"),
        ("led", "words ending in common suffixes", "Should be very low similarity"),
        ("sports scores", "numerical data in sports contexts", "Should be medium-high similarity"),
        ("Bob", "proper nouns, particularly names", "Should be high similarity"),
    ]
    
    print("SEMANTIC SIMILARITY TESTING")
    print("=" * 50)
    
    for golden, autogen, description in test_cases:
        results = evaluator.evaluate(golden, autogen)
        print(f"\nGolden: '{golden}'")
        print(f"Autogen: '{autogen}'")
        print(f"Expected: {description}")
        print(f"Word overlap: {results['word_overlap']:.3f}")
        print(f"Concept-based: {results['concept_based']:.3f}")
        if 'embedding' in results:
            print(f"Embedding: {results['embedding']:.3f}")
        print(f"Combined: {results['combined']:.3f}")

if __name__ == "__main__":
    test_semantic_evaluator()