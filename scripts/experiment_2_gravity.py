import torch
import numpy as np
from transformer_lens import HookedTransformer

# 1. SETUP
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

# 2. THE DATA (Same as before)
facts = [
    "The earth revolves around the sun",
    "Water is composed of hydrogen and oxygen",
    "Paris is the capital of France",
    "Two plus two equals four",
    "The sky is blue during the day",
    "Humans need oxygen to breathe",
    "Trees grow from the ground",
    "Fish live in water",
    "Fire is hot",
    "Ice is cold"
]

lies = [
    "The earth is flat and rests on a turtle",
    "Water is made of dry sand and dust",
    "Paris is the capital of Germany",
    "Two plus two equals ninety-nine",
    "The sky is neon green underground",
    "Humans can survive without breathing",
    "Trees fall upwards into the sky",
    "Fish fly in outer space",
    "Fire is colder than ice",
    "Ice burns like lava"
]

# 3. CALCULATE MASS (The Centroids)
def get_mass_center(sentences):
    vectors = []
    for s in sentences:
        _, cache = model.run_with_cache(s)
        # We grab the vector at Layer 10 this time (often better for "facts" than 11)
        # and we look at the LAST token.
        vec = cache["resid_post", 10][0, -1, :].cpu().detach().numpy()
        vectors.append(vec)
    
    # Calculate the average position (Center of Gravity)
    return np.mean(vectors, axis=0)

print("Calculating the Gravity of Truth...")
fact_center = get_mass_center(facts)
lie_center = get_mass_center(lies)

# 4. DEFINE THE AXIS (The "Truth Vector")
# This is the arrow that points from the Lie Valley to the Truth Ridge.
truth_direction = fact_center - lie_center

# 5. THE SCORING MECHANISM
def get_truth_score(statement):
    _, cache = model.run_with_cache(statement)
    vec = cache["resid_post", 10][0, -1, :].cpu().detach().numpy()
    
    # We use the "Dot Product". 
    # This asks: "How much does this statement's vector align with the Truth Direction?"
    # Positive score = Aligned with Truth. Negative score = Aligned with Lies.
    return np.dot(vec, truth_direction)

# 6. TEST
print("\n--- Testing the Gravity Model ---")

test_cases = [
    "The sun is a star",          # FACT
    "The moon is made of cheese", # LIE
    "Dogs are mammals",           # FACT
    "Dogs are reptiles"           # LIE
]

for stmt in test_cases:
    score = get_truth_score(stmt)
    print(f"Statement: '{stmt}'")
    print(f"Relational Tension: {score:.4f}")
    
    if score > 0:
        print("Verdict: RIDGE (Fact)")
    else:
        print("Verdict: VALLEY (Lie)")
    print("-" * 20)