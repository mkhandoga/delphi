#!/usr/bin/env python3
"""
Soft Prompt Optimization for Feature Labeling

Uses learnable soft tokens to optimize labeling quality without changing model parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np

class SoftPromptDataset(Dataset):
    """Dataset for training soft prompts on feature labeling task."""
    
    def __init__(self, golden_features: Dict, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for feature_key, golden_label in golden_features.items():
            # Mock feature activation examples - in practice, extract from your latent cache
            examples_text = f"Feature {feature_key} activates on: [example1] [example2] [example3]"
            
            self.data.append({
                'examples': examples_text,
                'target_label': golden_label,
                'feature_key': feature_key
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize examples
        examples_tokens = self.tokenizer(
            item['examples'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target label
        target_tokens = self.tokenizer(
            item['target_label'],
            max_length=50,  # Labels should be short
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': examples_tokens['input_ids'].squeeze(),
            'attention_mask': examples_tokens['attention_mask'].squeeze(),
            'target_ids': target_tokens['input_ids'].squeeze(),
            'target_mask': target_tokens['attention_mask'].squeeze(),
            'feature_key': item['feature_key']
        }

class SoftPromptModel(nn.Module):
    """Soft prompt model that prepends learnable tokens to the input."""
    
    def __init__(self, base_model, tokenizer, n_soft_tokens: int = 20):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.n_soft_tokens = n_soft_tokens
        
        # Get embedding dimension from base model
        self.embed_dim = base_model.config.hidden_size
        
        # Learnable soft prompt embeddings
        self.soft_embeddings = nn.Parameter(
            torch.randn(n_soft_tokens, self.embed_dim) * 0.1
        )
        
        # Initialize with good starting values
        self._initialize_soft_embeddings()
    
    def _initialize_soft_embeddings(self):
        """Initialize soft embeddings with semantically meaningful tokens."""
        
        # Initialize with embeddings of relevant words
        init_words = [
            "label", "feature", "concept", "pattern", "specific", 
            "precise", "concise", "identifies", "detects", "represents",
            "technical", "term", "name", "word", "token",
            "syntax", "semantic", "linguistic", "neural", "activation"
        ]
        
        word_embeddings = []
        for word in init_words[:self.n_soft_tokens]:
            # Get embedding of the word
            token_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
            embedding = self.base_model.get_input_embeddings()(torch.tensor([token_id]))
            word_embeddings.append(embedding.squeeze(0))
        
        # Pad if needed
        while len(word_embeddings) < self.n_soft_tokens:
            word_embeddings.append(torch.randn(self.embed_dim) * 0.1)
        
        # Initialize soft embeddings
        init_embeddings = torch.stack(word_embeddings)
        self.soft_embeddings.data = init_embeddings
    
    def forward(self, input_ids, attention_mask, target_ids=None):
        batch_size = input_ids.size(0)
        
        # Get input embeddings
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Expand soft embeddings for batch
        soft_embeds = self.soft_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate soft embeddings with input embeddings  
        full_embeddings = torch.cat([soft_embeds, input_embeddings], dim=1)
        
        # Extend attention mask for soft tokens
        soft_attention = torch.ones(batch_size, self.n_soft_tokens, device=attention_mask.device)
        full_attention_mask = torch.cat([soft_attention, attention_mask], dim=1)
        
        # Forward pass through base model
        outputs = self.base_model(
            inputs_embeds=full_embeddings,
            attention_mask=full_attention_mask,
            labels=target_ids
        )
        
        return outputs
    
    def generate_label(self, examples_text: str, max_length: int = 10):
        """Generate a label for given examples."""
        
        # Tokenize input
        inputs = self.tokenizer(
            examples_text + " Label:",
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        # Generate with soft prompt
        with torch.no_grad():
            batch_size = inputs['input_ids'].size(0)
            input_embeddings = self.base_model.get_input_embeddings()(inputs['input_ids'])
            
            # Add soft embeddings
            soft_embeds = self.soft_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            full_embeddings = torch.cat([soft_embeds, input_embeddings], dim=1)
            
            # Generate
            generated = self.base_model.generate(
                inputs_embeds=full_embeddings,
                attention_mask=torch.cat([
                    torch.ones(batch_size, self.n_soft_tokens),
                    inputs['attention_mask']
                ], dim=1),
                max_length=full_embeddings.size(1) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode only the generated part
            generated_text = self.tokenizer.decode(
                generated[0][full_embeddings.size(1):], 
                skip_special_tokens=True
            ).strip()
            
            return generated_text

def train_soft_prompt(
    model: SoftPromptModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.01
):
    """Train the soft prompt model."""
    
    # Only optimize soft embeddings, freeze base model
    optimizer = optim.AdamW([model.soft_embeddings], lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_ids=batch['target_ids']
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_([model.soft_embeddings], max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        val_loss = evaluate_soft_prompt(model, val_loader)
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.soft_embeddings, "best_soft_prompt.pt")
            print("  ✅ Saved new best model")
    
    return model

def evaluate_soft_prompt(model: SoftPromptModel, data_loader: DataLoader):
    """Evaluate soft prompt model."""
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'], 
                target_ids=batch['target_ids']
            )
            total_loss += outputs.loss.item()
    
    model.train()
    return total_loss / len(data_loader)

def create_soft_prompt_config() -> Dict:
    """Configuration for soft prompt optimization."""
    
    return {
        "optimization_method": "soft_prompting",
        "architecture": {
            "n_soft_tokens": 20,
            "initialization": "semantic_words",
            "embedding_dim": "auto",  # From base model
            "position": "prefix"  # Prepend to input
        },
        "training": {
            "epochs": 20,
            "learning_rate": 0.01,
            "batch_size": 8,
            "gradient_clipping": 1.0,
            "scheduler": "cosine_annealing",
            "early_stopping": {
                "patience": 5,
                "metric": "validation_loss"
            }
        },
        "evaluation": {
            "metrics": ["perplexity", "label_quality", "semantic_similarity"],
            "validation_split": 0.2
        },
        "expected_benefits": [
            "Parameter-efficient (only ~20K parameters)",
            "Preserves base model capabilities", 
            "Fast training and inference",
            "Transferable across similar tasks"
        ]
    }

def main():
    """Main soft prompting optimization pipeline."""
    
    print("Soft Prompt Optimization for Feature Labeling")
    print("=" * 45)
    
    # Configuration
    config = create_soft_prompt_config()
    
    print("📋 Soft Prompting Strategy:")
    print(f"- Soft tokens: {config['architecture']['n_soft_tokens']}")
    print(f"- Training epochs: {config['training']['epochs']}")
    print(f"- Learning rate: {config['training']['learning_rate']}")
    
    print("\n✨ Expected benefits:")
    for benefit in config['expected_benefits']:
        print(f"  • {benefit}")
    
    print("\n🔧 Implementation steps:")
    print("1. Load your base language model")
    print("2. Create SoftPromptModel wrapper")
    print("3. Prepare training data from golden labels")  
    print("4. Train soft embeddings (freeze base model)")
    print("5. Evaluate and save best checkpoint")
    print("6. Deploy in Delphi pipeline")
    
    # Save configuration
    with open("soft_prompt_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Saved configuration to soft_prompt_config.json")

if __name__ == "__main__":
    main()