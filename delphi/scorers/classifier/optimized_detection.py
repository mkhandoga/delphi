"""
Optimized detection scorer with multiple enhancement strategies.
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ...clients.client import Client
from ...latents import LatentRecord
from ...scorers.scorer import Scorer
from .classifier import Classifier
from .sample import Sample, examples_to_samples

logger = logging.getLogger(__name__)

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available. DSPy optimization will be disabled.")


class OptimizedDetectionScorer(Classifier, Scorer):
    """Enhanced detection scorer with multiple optimization strategies."""
    
    name = "optimized_detection"
    
    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        n_examples_shown: int = 5,
        log_prob: bool = True,
        temperature: float = 0.0,
        use_dspy: bool = True,
        use_ensemble: bool = True,
        use_dynamic_examples: bool = True,
        use_confidence_weighting: bool = True,
        confidence_threshold: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
        **generation_kwargs,
    ):
        super().__init__(
            client=client,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            log_prob=log_prob,
            temperature=temperature,
            **generation_kwargs,
        )
        
        self.use_dspy = use_dspy and DSPY_AVAILABLE
        self.use_ensemble = use_ensemble
        self.use_dynamic_examples = use_dynamic_examples
        self.use_confidence_weighting = use_confidence_weighting
        self.confidence_threshold = confidence_threshold
        
        # Initialize embedding model for similarity-based example selection
        if use_dynamic_examples:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                logger.warning(f"Could not load embedding model {embedding_model}: {e}")
                self.use_dynamic_examples = False
        
        # Initialize DSPy if available
        if self.use_dspy:
            self._init_dspy()
        
        # Example bank for dynamic selection
        self.example_bank = self._build_example_bank()
    
    def _init_dspy(self):
        """Initialize DSPy components."""
        if not DSPY_AVAILABLE:
            return
            
        class DetectionSignature(dspy.Signature):
            """Evaluate whether text examples match a latent feature explanation."""
            explanation = dspy.InputField(desc="Description of the latent feature")
            examples = dspy.InputField(desc="Text examples to classify")
            predictions = dspy.OutputField(desc="Binary predictions as list")
            reasoning = dspy.OutputField(desc="Step-by-step reasoning")
            confidence = dspy.OutputField(desc="Confidence scores for each prediction")
        
        class OptimizedDetector(dspy.Module):
            def __init__(self):
                self.detect = dspy.ChainOfThought(DetectionSignature)
            
            def forward(self, explanation, examples):
                return self.detect(explanation=explanation, examples=examples)
        
        self.dspy_detector = OptimizedDetector()
    
    def _build_example_bank(self) -> List[Dict[str, str]]:
        """Build a bank of examples for dynamic selection."""
        return [
            {
                "explanation": "Words related to American football positions, specifically the tight end position.",
                "examples": "Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ",
                "context": "sports football positions"
            },
            {
                "explanation": "The word \"guys\" in the phrase \"you guys\".",
                "examples": "I want to remind you all that 10 days ago",
                "context": "informal address pronouns"
            },
            {
                "explanation": "\"of\" before words that start with a capital letter.",
                "examples": "climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.",
                "context": "prepositions capitalization"
            }
        ]
    
    def _get_domain_specific_system_prompt(self) -> str:
        """Get enhanced system prompt with SAE domain knowledge."""
        return """You are an expert in sparse autoencoder (SAE) feature analysis and computational linguistics.

SAE latents represent specific patterns in neural network activations. Common types include:
- Syntactic patterns (e.g., "prepositions before nouns", "verb tenses")
- Semantic concepts (e.g., "sports terminology", "emotional words")
- Positional patterns (e.g., "tokens at sentence boundaries", "punctuation patterns")
- Morphological patterns (e.g., "plural forms", "comparative adjectives")

For each text example, determine if it contains the described latent pattern.

Consider these factors:
1. Token positions and context
2. Activation strength indicators
3. Pattern completeness vs. partial matches
4. Common false positives to avoid

Return your response as a valid Python list of 1s and 0s, where 1 means the example matches the pattern and 0 means it doesn't."""
    
    def _get_chain_of_thought_prompt(self, explanation: str, examples: str) -> str:
        """Generate chain-of-thought reasoning prompt."""
        return f"""Latent explanation: {explanation}

First, analyze the key pattern:
- What specific tokens/phrases should activate this latent?
- What contextual clues are important?
- Are there common false positives to avoid?

Then for each example, reason through:
1. Does it contain the target pattern?
2. Is the context appropriate?
3. How confident are you (0-1 scale)?
4. Final prediction: [1/0]

Examples:
{examples}

Provide your reasoning, then your final predictions as a Python list."""
    
    def _get_semantic_similarity_prompt(self, explanation: str, examples: str) -> str:
        """Generate semantic similarity focused prompt."""
        return f"""Focus on semantic similarity to identify pattern matches.

Latent pattern: {explanation}

For each example, evaluate semantic alignment with the pattern:
- Does the meaning match the described concept?
- Are there related semantic fields activated?
- Consider synonyms and conceptual overlap

Examples:
{examples}

Return predictions as a Python list [1,0,1,0,...]."""
    
    def _get_strict_pattern_prompt(self, explanation: str, examples: str) -> str:
        """Generate strict pattern matching prompt."""
        return f"""Apply strict pattern matching criteria.

Pattern to match: {explanation}

Look for exact or very close matches only:
- Must contain the specific tokens/structures described
- Context must be appropriate
- Avoid false positives from similar but different patterns

Examples:
{examples}

Return predictions as a Python list [1,0,1,0,...]."""
    
    def _select_dynamic_examples(self, explanation: str, k: int = 3) -> List[Dict[str, str]]:
        """Select most relevant examples based on semantic similarity."""
        if not self.use_dynamic_examples or not hasattr(self, 'embedding_model'):
            return self.example_bank[:k]
        
        try:
            # Embed the explanation
            explanation_emb = self.embedding_model.encode([explanation])
            
            # Embed all examples in the bank
            bank_explanations = [ex['explanation'] for ex in self.example_bank]
            bank_embeddings = self.embedding_model.encode(bank_explanations)
            
            # Calculate similarities
            similarities = cosine_similarity(explanation_emb, bank_embeddings)[0]
            
            # Select top-k most similar
            top_indices = np.argsort(similarities)[-k:][::-1]
            return [self.example_bank[i] for i in top_indices]
        
        except Exception as e:
            logger.warning(f"Dynamic example selection failed: {e}")
            return self.example_bank[:k]
    
    def _confidence_weighted_predictions(
        self, 
        predictions: List[bool], 
        confidences: List[float]
    ) -> Tuple[List[bool], List[float]]:
        """Apply confidence weighting to predictions."""
        if not self.use_confidence_weighting:
            return predictions, confidences
        
        reliable_indices = [
            i for i, conf in enumerate(confidences) 
            if conf >= self.confidence_threshold
        ]
        
        # If we have enough reliable predictions, use only those
        if len(reliable_indices) >= len(predictions) * 0.5:
            weighted_preds = [predictions[i] for i in reliable_indices]
            weighted_confs = [confidences[i] for i in reliable_indices]
        else:
            # Use all predictions but log uncertainty
            logger.info(f"Low confidence predictions: {len(reliable_indices)}/{len(predictions)} above threshold")
            weighted_preds = predictions
            weighted_confs = confidences
        
        return weighted_preds, weighted_confs
    
    async def _ensemble_predict(
        self, 
        explanation: str, 
        examples: str
    ) -> Tuple[List[bool], List[float]]:
        """Use ensemble of different prompt strategies."""
        if not self.use_ensemble:
            # Fall back to single best approach
            return await self._single_predict(explanation, examples, "cot")
        
        approaches = [
            ("cot", self._get_chain_of_thought_prompt),
            ("semantic", self._get_semantic_similarity_prompt),
            ("strict", self._get_strict_pattern_prompt)
        ]
        
        all_predictions = []
        all_confidences = []
        
        for name, prompt_func in approaches:
            try:
                predictions, confidences = await self._single_predict(
                    explanation, examples, name, prompt_func
                )
                all_predictions.append(predictions)
                all_confidences.append(confidences)
            except Exception as e:
                logger.warning(f"Ensemble approach '{name}' failed: {e}")
        
        if not all_predictions:
            raise RuntimeError("All ensemble approaches failed")
        
        # Majority vote with confidence weighting
        final_predictions = []
        final_confidences = []
        
        for i in range(len(all_predictions[0])):
            votes = [preds[i] for preds in all_predictions if i < len(preds)]
            confs = [confs[i] for confs in all_confidences if i < len(confs)]
            
            if not votes:
                final_predictions.append(False)
                final_confidences.append(0.0)
                continue
            
            # Weighted majority vote
            weighted_sum = sum(v * c for v, c in zip(votes, confs))
            total_weight = sum(confs)
            
            if total_weight > 0:
                final_pred = weighted_sum / total_weight > 0.5
                final_conf = total_weight / len(votes)  # Average confidence
            else:
                final_pred = sum(votes) > len(votes) / 2  # Simple majority
                final_conf = 0.5
            
            final_predictions.append(final_pred)
            final_confidences.append(final_conf)
        
        return final_predictions, final_confidences
    
    async def _single_predict(
        self, 
        explanation: str, 
        examples: str, 
        approach: str = "cot",
        prompt_func = None
    ) -> Tuple[List[bool], List[float]]:
        """Single prediction approach."""
        if prompt_func is None:
            prompt_func = self._get_chain_of_thought_prompt
        
        # Build prompt
        system_prompt = self._get_domain_specific_system_prompt()
        user_content = prompt_func(explanation, examples)
        
        # Select dynamic examples if enabled
        if self.use_dynamic_examples:
            few_shot_examples = self._select_dynamic_examples(explanation)
        else:
            few_shot_examples = self.example_bank[:3]
        
        # Build full prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add few-shot examples
        for ex in few_shot_examples:
            messages.append({
                "role": "user", 
                "content": f"Latent explanation: {ex['explanation']}\n\nExamples:\n{ex['examples']}"
            })
            messages.append({
                "role": "assistant", 
                "content": "[1]"  # Simplified for demo
            })
        
        messages.append({"role": "user", "content": user_content})
        
        # Generate response
        response = await self.client.generate(messages, **self.generation_kwargs)
        
        if response is None:
            raise RuntimeError("Failed to get response from client")
        
        # Parse predictions
        predictions, confidences = self._parse_response(response.text)
        
        return predictions, confidences
    
    def _parse_response(self, response_text: str) -> Tuple[List[bool], List[float]]:
        """Parse response to extract predictions and confidence scores."""
        # Extract predictions list
        pattern = r"\[.*?\]"
        match = re.search(pattern, response_text)
        if match is None:
            raise ValueError("No predictions list found in response")
        
        predictions = json.loads(match.group(0))
        
        # Extract confidence scores if present
        confidence_pattern = r"confidence[:\s]*\[([0-9.,\s]+)\]"
        conf_match = re.search(confidence_pattern, response_text, re.IGNORECASE)
        
        if conf_match:
            try:
                confidences = json.loads(f"[{conf_match.group(1)}]")
            except:
                confidences = [0.8] * len(predictions)  # Default confidence
        else:
            confidences = [0.8] * len(predictions)  # Default confidence
        
        return predictions, confidences
    
    async def _generate(self, explanation: str, batch: List[Sample]) -> List:
        """Generate predictions for a batch of samples."""
        examples = "\n".join(
            f"Example {i}: {sample.text}" for i, sample in enumerate(batch)
        )
        
        try:
            if self.use_dspy and hasattr(self, 'dspy_detector'):
                # Use DSPy optimized approach
                result = self.dspy_detector(explanation=explanation, examples=examples)
                predictions = json.loads(result.predictions)
                confidences = json.loads(result.confidence) if hasattr(result, 'confidence') else [0.8] * len(predictions)
            else:
                # Use ensemble or single approach
                predictions, confidences = await self._ensemble_predict(explanation, examples)
            
            # Apply confidence weighting
            predictions, confidences = self._confidence_weighted_predictions(predictions, confidences)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {repr(e)}")
            predictions = [None] * len(batch)
            confidences = [None] * len(batch)
        
        # Build results
        results = []
        for sample, prediction, confidence in zip(batch, predictions, confidences):
            result = sample.data
            result.prediction = bool(prediction) if prediction is not None else None
            if prediction is not None:
                result.correct = prediction == result.activating
            else:
                result.correct = None
            result.probability = confidence
            results.append(result)
            
            if self.verbose:
                logger.info(
                    f"Example: {sample.text[:50]}..., "
                    f"Prediction: {prediction}, "
                    f"Confidence: {confidence:.3f}"
                )
        
        return results
    
    def prompt(self, examples: str, explanation: str) -> List[Dict]:
        """Build prompt for compatibility with base class."""
        system_prompt = self._get_domain_specific_system_prompt()
        user_content = self._get_chain_of_thought_prompt(explanation, examples)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def _prepare(self, record: LatentRecord) -> List[Sample]:
        """Prepare samples for classification."""
        if len(record.not_active) > 0:
            samples = examples_to_samples(record.not_active)
        else:
            samples = []
        
        samples.extend(examples_to_samples(record.test))
        return samples


class AdaptiveLearningDetector(OptimizedDetectionScorer):
    """Extension with active learning capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertain_examples = []
        self.performance_history = {}
    
    async def _generate_with_uncertainty_tracking(
        self, 
        explanation: str, 
        batch: List[Sample]
    ) -> List:
        """Generate predictions while tracking uncertain cases."""
        results = await super()._generate(explanation, batch)
        
        # Track uncertain cases for potential human labeling
        for result in results:
            if result.probability and result.probability < 0.8:
                self.uncertain_examples.append({
                    'text': result.text if hasattr(result, 'text') else str(result),
                    'explanation': explanation,
                    'prediction': result.prediction,
                    'confidence': result.probability
                })
        
        return results
    
    def get_uncertain_examples(self, n: int = 10) -> List[Dict]:
        """Get the most uncertain examples for human labeling."""
        sorted_examples = sorted(
            self.uncertain_examples, 
            key=lambda x: x['confidence']
        )
        return sorted_examples[:n]