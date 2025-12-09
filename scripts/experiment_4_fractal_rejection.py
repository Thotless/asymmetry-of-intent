"""
TERMINOLOGY MAPPING:
This script uses the philosophical terminology defined in "The Asymmetry of Intent."
For engineers reviewing the code, here is the translation key:

- "Rejection Vector"        -> Negative steering vector / Difference in means
- "Fractal Steering"        -> Hierarchical activation injection (Multi-layer)
- "Texture"                 -> The geometric direction of a concept (e.g., Lying)
- "Mending"                 -> Inference-time intervention
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer

# 1. SETUP
print("Initializing Fractal Rejection System...")
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

# 2. DEFINE THE DATASETS (The "Texture" Definitions)
# We don't list every lie; we define the *direction* of the error.

# Level 1: Coherence (English vs. Chaos)
coherence_good = [
    "The quick brown fox jumps over the lazy dog",
    "Please wait for the signal before crossing",
    "The weather today is sunny and clear",
    "I enjoy reading books about history"
]
coherence_bad = [
    "Fox brown quick the dog lazy over jumps",
    "Signal crossing before wait please the",
    "Sunny clear is today weather the and",
    "History books reading enjoy I about"
]

# Level 2: Logic (Sense vs. Non-Sequitur)
logic_good = [
    "I dropped the glass and it shattered",
    "Because it was raining, I took an umbrella",
    "She was hungry so she ate a sandwich",
    "The car ran out of gas so it stopped"
]
logic_bad = [
    "I dropped the glass and it flew to Mars",
    "Because it was raining, I ate a brick",
    "She was hungry so she painted a fence",
    "The car ran out of gas so it turned into a cat"
]

# Level 3: Fact (Truth vs. Hallucination) - From Experiment 2
fact_good = [
    "The earth revolves around the sun",
    "Water is composed of hydrogen and oxygen",
    "Dogs are mammals",
    "Paris is the capital of France"
]
fact_bad = [
    "The earth revolves around the moon",
    "Water is composed of sand and dust",
    "Dogs are reptiles",
    "Paris is the capital of Germany"
]

# 3. CALCULATE THE VECTORS
def get_vector(good_list, bad_list, layer):
    """Calculates the geometric direction from 'Bad' to 'Good' at a specific layer."""
    good_vecs = []
    bad_vecs = []
    
    for s in good_list:
        _, cache = model.run_with_cache(s)
        # Use the last token's residual stream
        good_vecs.append(cache["resid_pre", layer][0, -1, :].cpu().detach())
        
    for s in bad_list:
        _, cache = model.run_with_cache(s)
        bad_vecs.append(cache["resid_pre", layer][0, -1, :].cpu().detach())
        
    # Vector = Average(Good) - Average(Bad)
    center_good = torch.stack(good_vecs).mean(dim=0)
    center_bad = torch.stack(bad_vecs).mean(dim=0)
    
    vec = center_good - center_bad
    return vec / vec.norm() # Normalize

print("Calculating Fractal Vectors...")
# We place filters at different depths to catch different types of noise
vec_coherence = get_vector(coherence_good, coherence_bad, layer=4)
vec_logic     = get_vector(logic_good, logic_bad, layer=7)
vec_fact      = get_vector(fact_good, fact_bad, layer=10)

# 4. DEFINE THE FRACTAL HOOKS
# Each hook applies its specific rejection vector to its specific layer.

def hook_coherence(resid_pre, hook):
    # Strength 5.0: Just a gentle nudge to stay coherent
    return resid_pre + (vec_coherence.to(resid_pre.device) * 5.0)

def hook_logic(resid_pre, hook):
    # Strength 8.0: Stronger push to ensure logical flow
    return resid_pre + (vec_logic.to(resid_pre.device) * 8.0)

def hook_fact(resid_pre, hook):
    # Strength 15.0: Massive force to reject the "Lie Texture"
    return resid_pre + (vec_fact.to(resid_pre.device) * 15.0)

# 5. THE TEST
# A prompt that invites all three failures: Nonsense, Dream-logic, and Lies.
test_prompt = "The scientist discovered that the moon is made of"

# Generate CONTROL (No Filters)
print(f"\nPrompt: '{test_prompt}'")
print("-" * 40)
print("1. Control Generation (Natural Drift):")
output_control = model.generate(test_prompt, max_new_tokens=20, verbose=False)
print(f"   > {output_control}")

# Generate EXPERIMENTAL (Fractal Filters)
# We apply all three hooks simultaneously at their respective layers.
hooks = [
    ("blocks.4.hook_resid_pre", hook_coherence),
    ("blocks.7.hook_resid_pre", hook_logic),
    ("blocks.10.hook_resid_pre", hook_fact)
]

print("\n2. Fractal Generation (Hierarchical Steering):")
with model.hooks(fwd_hooks=hooks):
    output_fractal = model.generate(test_prompt, max_new_tokens=20, verbose=False)
print(f"   > {output_fractal}")

print("-" * 40)
print("Observation: Does the Fractal Generation feel 'tighter' and more grounded?")