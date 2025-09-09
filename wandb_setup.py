#!/usr/bin/env python3
"""
Wandb Setup Helper for Delphi Prompt Optimization

Sets up Weights & Biases tracking for systematic prompt experiments.
"""

import os
import subprocess
from pathlib import Path

def check_wandb_installation():
    """Check if wandb is installed and working."""
    try:
        import wandb
        return True, wandb.__version__
    except ImportError:
        return False, None

def install_wandb():
    """Install wandb if not present."""
    print("📦 Installing wandb...")
    try:
        subprocess.run(["pip", "install", "wandb"], check=True)
        print("✅ Wandb installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install wandb")
        return False

def setup_wandb_auth():
    """Setup wandb authentication."""
    print("\n🔐 Setting up Wandb authentication...")
    
    # Check if already authenticated
    try:
        import wandb
        if wandb.api.api_key:
            print("✅ Already authenticated with wandb")
            return True
    except:
        pass
    
    print("Options:")
    print("1. Run 'wandb login' in terminal (interactive)")
    print("2. Set WANDB_API_KEY environment variable")
    print("3. Create .wandb_api_key file")
    
    choice = input("\nHow would you like to authenticate? (1/2/3): ").strip()
    
    if choice == "1":
        print("Run this command in your terminal:")
        print("  wandb login")
        return False
    
    elif choice == "2":
        api_key = input("Enter your WANDB_API_KEY: ").strip()
        if api_key:
            print(f"Add this to your environment:")
            print(f"  export WANDB_API_KEY={api_key}")
            print("Or add it to your .bashrc/.zshrc")
            return False
    
    elif choice == "3":
        api_key = input("Enter your WANDB_API_KEY: ").strip()
        if api_key:
            wandb_key_file = Path.home() / ".wandb_api_key"
            wandb_key_file.write_text(api_key)
            print(f"✅ API key saved to {wandb_key_file}")
            return True
    
    return False

def create_wandb_config():
    """Create wandb configuration for the project."""
    config = {
        "project": "delphi-prompt-optimization",
        "entity": None,  # Will use default entity
        "tags": ["prompt-optimization", "interpretability", "delphi"],
        "notes": "Systematic optimization of prompts for latent feature labeling"
    }
    
    # Get user input for entity
    entity = input(f"\nEnter your wandb entity/username (optional): ").strip()
    if entity:
        config["entity"] = entity
    
    # Save config
    config_file = Path("wandb_config.json")
    import json
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Wandb config saved to {config_file}")
    return config

def test_wandb_integration():
    """Test wandb integration with a dummy experiment."""
    print("\n🧪 Testing wandb integration...")
    
    try:
        from experiment_tracker import ExperimentTracker
        
        # Create tracker with wandb enabled
        tracker = ExperimentTracker(use_wandb=True)
        
        if not tracker.use_wandb:
            print("❌ Wandb integration not working")
            return False
        
        print("✅ Wandb integration working!")
        print(f"   Project: {tracker.wandb_project}")
        print(f"   Entity: {tracker.wandb_entity or 'default'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing wandb: {e}")
        return False

def main():
    print("🚀 WANDB SETUP FOR DELPHI PROMPT OPTIMIZATION")
    print("=" * 50)
    
    # Step 1: Check/install wandb
    is_installed, version = check_wandb_installation()
    if not is_installed:
        print("❌ Wandb not installed")
        if input("Install wandb now? (y/n): ").lower().startswith('y'):
            if not install_wandb():
                return
        else:
            print("Wandb is required for enhanced experiment tracking")
            return
    else:
        print(f"✅ Wandb installed (version {version})")
    
    # Step 2: Setup authentication
    if not setup_wandb_auth():
        print("\n⚠️  Authentication not completed.")
        print("   Complete authentication and run this script again.")
        return
    
    # Step 3: Create project config
    config = create_wandb_config()
    
    # Step 4: Test integration
    if test_wandb_integration():
        print("\n🎉 WANDB SETUP COMPLETE!")
        print("\n📋 NEXT STEPS:")
        print("1. Run baseline experiment:")
        print("   python prompt_experiment_workflow.py --baseline")
        print("\n2. View experiments on wandb dashboard:")
        print(f"   https://wandb.ai/{config.get('entity', 'your-entity')}/{config['project']}")
        print("\n3. Make prompt changes and track improvements systematically")
        
    else:
        print("\n❌ Setup incomplete. Check wandb authentication and try again.")

if __name__ == "__main__":
    main()