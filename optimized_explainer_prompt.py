"""
Optimized explainer prompt based on golden label analysis.

This replaces the verbose current prompt with one that produces concise, accurate labels.
"""

# Get some good examples from golden labels
GOLDEN_EXAMPLES = """
Examples of EXCELLENT labels (study these):
- "facts" (not "statements presenting factual information")
- "names" (not "proper nouns and entities in text") 
- "colon punctuation" (not "punctuation marks that introduce lists")
- "sports scores" (not "numerical data in sports contexts")
- "code syntax" (not "programming language elements")
- "city names" (not "geographical location entities")
"""

OPTIMIZED_SYSTEM_PROMPT = f"""You are a precise neural network feature labeling expert. Your task is to create a concise, specific label for what this feature detects.

{GOLDEN_EXAMPLES}

CRITICAL RULES:
- Maximum 3 words
- Be specific, not generic (avoid "various", "common", "often", "specific", "different") 
- Focus on the exact concept, not descriptions
- Use technical terms when appropriate

BAD examples (never do this):
- "Various words and tokens that often appear in text"
- "Common patterns frequently seen in contexts" 
- "Words that are contextually significant"
- "Text elements with particular characteristics"

You will be given text examples with important tokens marked between <<delimiters>>. 

Your response MUST end with: [EXPLANATION]: [your 1-3 word label]

Guidelines:
- Simply describe what concept the marked tokens represent
- Keep explanations SHORT and concise
- Do not make lists of possible explanations
- Do not mention the delimiter tokens (<< >>)

{{prompt}}
"""

def create_optimized_prompt():
    """Returns the optimized system prompt to replace the current verbose one."""
    return OPTIMIZED_SYSTEM_PROMPT

# Updated examples that demonstrate conciseness
OPTIMIZED_EXAMPLE_1_EXPLANATION = """
[EXPLANATION]: Common idioms
"""

OPTIMIZED_EXAMPLE_2_EXPLANATION = """
[EXPLANATION]: Comparative suffix "er"
"""

OPTIMIZED_EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Container nouns in speech
"""

def get_optimized_examples():
    """Returns optimized example responses that are much more concise."""
    return {
        'EXAMPLE_1_EXPLANATION': OPTIMIZED_EXAMPLE_1_EXPLANATION,
        'EXAMPLE_2_EXPLANATION': OPTIMIZED_EXAMPLE_2_EXPLANATION, 
        'EXAMPLE_3_EXPLANATION': OPTIMIZED_EXAMPLE_3_EXPLANATION,
    }

# Integration instructions
INTEGRATION_STEPS = """
To integrate this optimized prompt:

1. BACKUP your current prompts.py file
2. Replace the SYSTEM prompt (line 26) with the optimized version
3. Optionally replace the example explanations with shorter versions
4. Test on a few features to see the improvement
5. Run full evaluation to measure results

Expected improvements:
- 80%+ reduction in label length
- Higher semantic similarity with golden labels  
- Fewer generic terms
- More specific, actionable labels
"""

if __name__ == "__main__":
    print("OPTIMIZED EXPLAINER PROMPT")
    print("=" * 40)
    print(create_optimized_prompt())
    print("\n" + INTEGRATION_STEPS)