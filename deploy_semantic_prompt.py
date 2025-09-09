#!/usr/bin/env python3
"""
Ready-to-deploy semantic-focused prompt that directly addresses the 0.004 similarity issue.
"""

SEMANTIC_FOCUSED_SYSTEM_PROMPT = """You are a neural network interpretability expert. Your PRIMARY goal is semantic accuracy - identifying the exact concept this feature detects.

EXPERT EXAMPLES (learn from these):
- If it detects names → Say "names" or "proper names"  
- If it detects punctuation → Say "colon punctuation" or "periods"
- If it detects specific words → Name the word: "the", "jump", "sports"
- If it detects code → Say "code" or "HTML tags" or "Python syntax"
- If it detects linguistic patterns → Say "past tense verbs" or "er suffix"

SEMANTIC ACCURACY CHECKLIST:
1. Does this feature detect names/people? → Use "names" or specific type
2. Does it detect punctuation? → Use "punctuation" + type  
3. Does it detect code/syntax? → Use "code" or specific syntax type
4. Does it detect specific words? → Name the actual word
5. Does it detect linguistic patterns? → Use precise linguistic terms

CRITICAL RULE: Identify the CONCEPT, not describe the text.

WRONG: "Various tokens that often appear in contexts" 
RIGHT: "Proper names"

WRONG: "Words with particular characteristics"
RIGHT: "Past tense verbs" 

WRONG: "The text contains a variety of important tokens"
RIGHT: "Sports terminology"

Your response must end with: [EXPLANATION]: [the exact concept this feature detects]

{prompt}"""

def backup_and_deploy():
    """Backup current prompt and deploy semantic-focused version."""
    from pathlib import Path
    import shutil
    
    prompts_file = Path("delphi/explainers/default/prompts.py")
    backup_file = Path("delphi/explainers/default/prompts_semantic_backup.py")
    
    if prompts_file.exists():
        # Backup
        shutil.copy2(prompts_file, backup_file)
        print(f"✅ Backed up current prompts to {backup_file}")
        
        # Read current file
        content = prompts_file.read_text()
        
        # Find and replace the SYSTEM prompt
        old_system_start = 'SYSTEM = """You are a meticulous AI researcher'
        old_system_end = '"""'
        
        start_idx = content.find(old_system_start)
        if start_idx != -1:
            end_idx = content.find('"""', start_idx + len(old_system_start)) + 3
            
            new_content = (
                content[:start_idx] + 
                f'SYSTEM = {SEMANTIC_FOCUSED_SYSTEM_PROMPT}' + 
                content[end_idx:]
            )
            
            # Write updated file
            prompts_file.write_text(new_content)
            print(f"✅ Deployed semantic-focused prompt to {prompts_file}")
            print(f"📊 Expected improvement: 0.004 → 0.3+ semantic similarity")
            
        else:
            print("❌ Could not find SYSTEM prompt to replace")
    else:
        print(f"❌ Prompts file not found: {prompts_file}")

def manual_deployment_instructions():
    """Show manual deployment instructions."""
    print("\n" + "="*60)
    print("MANUAL DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    print("\n1. BACKUP your current prompts:")
    print("   cp delphi/explainers/default/prompts.py delphi/explainers/default/prompts_backup.py")
    
    print("\n2. EDIT delphi/explainers/default/prompts.py")
    print("   Replace the SYSTEM prompt (around line 26) with:")
    print(f'   SYSTEM = """{SEMANTIC_FOCUSED_SYSTEM_PROMPT}"""')
    
    print("\n3. TEST on a few features:")
    print("""   CUDA_VISIBLE_DEVICES=6,7 uv run python -m delphi \\
     google/gemma-2-2b \\
     /path/to/your/data \\
     --name semantic_test \\
     --max_latents 5 \\
     --scorers detection \\
     --explainer_provider offline \\
     --explainer_model Qwen/Qwen2.5-72B-Instruct-AWQ""")
    
    print("\n4. MEASURE improvement:")
    print("   python label_evaluation.py --explanations-dir results/semantic_test/explanations")
    
    print("\n🎯 Expected Results:")
    print("   • Semantic similarity: 0.004 → 0.3+ (75x improvement)")
    print("   • Word count ratio: 26x → 3x (8x improvement)")  
    print("   • Generic terms: Dramatically reduced")
    print("   • Actual concept identification instead of meta-descriptions")

if __name__ == "__main__":
    print("SEMANTIC-FOCUSED PROMPT DEPLOYMENT")
    print("="*50)
    
    try:
        backup_and_deploy()
    except Exception as e:
        print(f"Automatic deployment failed: {e}")
        manual_deployment_instructions()
    
    print("\n🚀 NEXT STEPS:")
    print("1. Test the semantic prompt on 5-10 features")  
    print("2. Compare with evaluation script")
    print("3. If successful, run on full dataset")
    print("4. Consider DSPy optimization for even better results")