"""
Tests for the OptimizedDetectionScorer.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
import json

import sys
sys.path.insert(0, '/mnt/ssd-1/soar-automated_interpretability/graphs/mykola/delphi')

from delphi.clients.client import Client, Response
from delphi.latents import LatentRecord, Latent, ActivatingExample, NonActivatingExample
from delphi.scorers.classifier.optimized_detection import (
    OptimizedDetectionScorer,
    AdaptiveLearningDetector
)


class MockClient(Client):
    """Mock client for testing."""
    
    def __init__(self, response_text="[1,0,1,0,1]"):
        super().__init__(model="test_model")
        self.response_text = response_text
    
    async def generate(self, prompt, **kwargs):
        return Response(text=self.response_text, logprobs=None)


@pytest.fixture
def sample_record():
    """Create a sample latent record for testing."""
    import torch
    
    # Create a sample latent
    latent = Latent(module_name="test_module", latent_index=123)
    
    # Create mock tensors
    mock_tokens = torch.randint(0, 1000, (5,))  # 5 tokens
    mock_activations = torch.randn(5)
    
    # Create activating examples with required fields
    test_examples = [
        ActivatingExample(
            tokens=mock_tokens,
            activations=mock_activations,
            str_tokens=["Patriots", "tight", "end", "Rob", "Gronkowski"]
        )
    ]
    
    # Create non-activating examples  
    not_active_examples = [
        NonActivatingExample(
            tokens=mock_tokens,
            activations=mock_activations,
            distance=0.5,
            str_tokens=["test", "tokens", "here"]
        )
    ]
    
    return LatentRecord(
        latent=latent,
        explanation="Test pattern for football positions",
        test=test_examples,
        not_active=not_active_examples
    )


@pytest.fixture
def mock_client():
    """Create a mock client."""
    return MockClient()


class TestOptimizedDetectionScorer:
    """Test suite for OptimizedDetectionScorer."""
    
    @pytest.mark.asyncio
    async def test_basic_initialization(self, mock_client):
        """Test basic scorer initialization."""
        scorer = OptimizedDetectionScorer(
            client=mock_client,
            verbose=False
        )
        
        assert scorer.name == "optimized_detection"
        assert scorer.client == mock_client
        assert not scorer.verbose
    
    @pytest.mark.asyncio
    async def test_domain_specific_prompt(self, mock_client):
        """Test domain-specific system prompt generation."""
        scorer = OptimizedDetectionScorer(client=mock_client)
        
        prompt = scorer._get_domain_specific_system_prompt()
        
        assert "sparse autoencoder" in prompt.lower()
        assert "syntactic patterns" in prompt.lower()
        assert "semantic concepts" in prompt.lower()
        assert "positional patterns" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_chain_of_thought_prompt(self, mock_client):
        """Test chain-of-thought prompt generation."""
        scorer = OptimizedDetectionScorer(client=mock_client)
        
        explanation = "Test pattern"
        examples = "Example 1: test\nExample 2: another test"
        
        prompt = scorer._get_chain_of_thought_prompt(explanation, examples)
        
        assert explanation in prompt
        assert examples in prompt
        assert "analyze the key pattern" in prompt.lower()
        assert "reason through" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_confidence_weighting(self, mock_client):
        """Test confidence weighting functionality."""
        scorer = OptimizedDetectionScorer(
            client=mock_client,
            use_confidence_weighting=True,
            confidence_threshold=0.7
        )
        
        predictions = [True, False, True, False]
        confidences = [0.9, 0.5, 0.8, 0.6]  # Only 2 out of 4 above 0.7 threshold
        
        weighted_preds, weighted_confs = scorer._confidence_weighted_predictions(
            predictions, confidences
        )
        
        # Should keep only high confidence predictions (2 out of 4)
        assert len(weighted_preds) == 2
        assert len(weighted_confs) == 2
        assert weighted_preds == [True, True]  # The high confidence ones
        assert weighted_confs == [0.9, 0.8]
    
    @pytest.mark.asyncio 
    async def test_response_parsing(self, mock_client):
        """Test response parsing for predictions and confidence."""
        scorer = OptimizedDetectionScorer(client=mock_client)
        
        # Test basic prediction parsing
        response_text = "Here are my predictions: [1,0,1,0] with high confidence."
        predictions, confidences = scorer._parse_response(response_text)
        
        assert predictions == [1, 0, 1, 0]
        assert len(confidences) == 4
        assert all(c == 0.8 for c in confidences)  # Default confidence
        
        # Test with confidence scores
        response_with_conf = "Predictions: [1,0,1] confidence: [0.9,0.3,0.8]"
        predictions, confidences = scorer._parse_response(response_with_conf)
        
        assert predictions == [1, 0, 1]
        assert confidences == [0.9, 0.3, 0.8]
    
    @pytest.mark.asyncio
    async def test_basic_scoring(self, mock_client, sample_record):
        """Test basic scoring functionality."""
        scorer = OptimizedDetectionScorer(
            client=mock_client,
            verbose=False,
            use_ensemble=False,
            use_dspy=False
        )
        
        result = await scorer(sample_record)
        
        assert result.record == sample_record
        assert len(result.score) > 0
        
        # Check that predictions were made
        for score in result.score:
            assert score.prediction is not None
            assert score.probability is not None
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_record):
        """Test error handling in generation."""
        # Create client that raises exception
        class FailingClient(Client):
            def __init__(self):
                super().__init__(model="test_model")
                
            async def generate(self, prompt, **kwargs):
                raise Exception("Test error")
        
        scorer = OptimizedDetectionScorer(
            client=FailingClient(),
            verbose=False
        )
        
        result = await scorer(sample_record)
        
        # Should handle errors gracefully
        assert len(result.score) > 0
        for score in result.score:
            assert score.prediction is None
            assert score.correct is None
    
    @pytest.mark.asyncio
    @patch('delphi.scorers.classifier.optimized_detection.SentenceTransformer')
    async def test_dynamic_example_selection(self, mock_transformer, mock_client):
        """Test dynamic example selection."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model
        
        scorer = OptimizedDetectionScorer(
            client=mock_client,
            use_dynamic_examples=True
        )
        
        examples = scorer._select_dynamic_examples("football positions", k=2)
        
        assert len(examples) <= 2
        assert all("explanation" in ex for ex in examples)
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, mock_client):
        """Test ensemble prediction functionality."""
        scorer = OptimizedDetectionScorer(
            client=mock_client,
            use_ensemble=True
        )
        
        explanation = "Test pattern"
        examples = "Example 1: test\nExample 2: another"
        
        # This should not raise an exception
        predictions, confidences = await scorer._ensemble_predict(explanation, examples)
        
        assert isinstance(predictions, list)
        assert isinstance(confidences, list)
        assert len(predictions) == len(confidences)


class TestAdaptiveLearningDetector:
    """Test suite for AdaptiveLearningDetector."""
    
    @pytest.mark.asyncio
    async def test_uncertainty_tracking(self, mock_client, sample_record):
        """Test uncertainty tracking functionality."""
        # Create client that returns low confidence predictions
        client_with_low_conf = MockClient("[1,0,1,0,1]")
        
        detector = AdaptiveLearningDetector(
            client=client_with_low_conf,
            verbose=False
        )
        
        # Mock low confidence in results
        with patch.object(detector, '_generate') as mock_generate:
            mock_result = Mock()
            mock_result.probability = 0.3  # Low confidence
            mock_result.prediction = True
            mock_generate.return_value = [mock_result]
            
            await detector(sample_record)
        
        # Should have tracked uncertain examples
        uncertain = detector.get_uncertain_examples(n=5)
        assert isinstance(uncertain, list)
    
    @pytest.mark.asyncio
    async def test_get_uncertain_examples(self, mock_client):
        """Test getting uncertain examples for human labeling."""
        detector = AdaptiveLearningDetector(client=mock_client)
        
        # Add some mock uncertain examples
        detector.uncertain_examples = [
            {'text': 'test1', 'confidence': 0.3, 'prediction': True, 'explanation': 'test'},
            {'text': 'test2', 'confidence': 0.7, 'prediction': False, 'explanation': 'test'},
            {'text': 'test3', 'confidence': 0.1, 'prediction': True, 'explanation': 'test'},
        ]
        
        uncertain = detector.get_uncertain_examples(n=2)
        
        assert len(uncertain) == 2
        # Should be sorted by confidence (lowest first)
        assert uncertain[0]['confidence'] <= uncertain[1]['confidence']


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_record):
        """Test the full optimization pipeline."""
        # Use a more realistic mock client
        class RealisticClient(Client):
            def __init__(self):
                super().__init__(model="test_model")
                
            async def generate(self, prompt, **kwargs):
                # Return different responses based on prompt content
                if "chain" in str(prompt).lower():
                    return Response(text="After reasoning: [1,0,1,0,1]", logprobs=None)
                else:
                    return Response(text="[1,0,1,0,1]", logprobs=None)
        
        client = RealisticClient()
        
        scorer = OptimizedDetectionScorer(
            client=client,
            verbose=False,
            use_ensemble=True,
            use_confidence_weighting=True,
            use_dynamic_examples=False,  # Disable to avoid embedding model issues
            use_dspy=False  # Disable DSPy to avoid import issues in tests
        )
        
        result = await scorer(sample_record)
        
        assert result is not None
        assert len(result.score) > 0
        assert all(score.prediction is not None for score in result.score)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])