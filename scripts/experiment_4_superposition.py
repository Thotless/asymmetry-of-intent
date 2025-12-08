import torch
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformer

# 1. SETUP
print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

# 2. TARGET THE SUPERPOSITION NEURON
# We confirmed 179 is the "Frankenstein" neuron
layer = 0
neuron_index = 179 

def get_neuron_activation(text):
    _, cache = model.run_with_cache(text)
    return cache["post", layer][0, :, neuron_index].max().item()

# 3. THE STIMULATION TEST
# CRITICAL FIX: We added the "Ambiguous" category (Web, Cloud, etc.)
# These are the words that should trigger the "Gold Bar" effect in your essay.
categories = {
    "Technology": [" computer", " internet", " software", " code", " server"],
    "Nature":     [" tree", " flower", " river", " mountain", " forest"],
    "Ambiguous":  [" web", " cloud", " bug", " mouse", " shell"], # The "Superposition" words
    "Grammar":    [" the", " and", " is", " of", " to"] # Control
}

print(f"\nGenerating Proof for Layer {layer}, Neuron {neuron_index}...")
results = []

for cat, words in categories.items():
    print(f"Testing Category: {cat}")
    for w in words:
        score = get_neuron_activation(w)
        print(f"  Word: '{w}' -> Activation: {score:.4f}")
        results.append({"Category": cat, "Word": w, "Activation": score})

# 4. VISUALIZE THE OVERLAP
print("\nBuilding Graph...")
df = pd.DataFrame(results)

# We force the colors to match your essay description:
# Tech = Blue, Nature = Red, Ambiguous = GOLD
color_map = {
    "Technology": "#636EFA", # Blue
    "Nature":     "#EF553B", # Red
    "Ambiguous":  "#FFD700", # GOLD (The Superposition Color)
    "Grammar":    "#7F7F7F"  # Grey
}

fig = px.bar(
    df, x="Word", y="Activation", color="Category",
    title=f"Neuron {neuron_index}: The Kaleidoscope Effect (Superposition)",
    labels={"Activation": "Firing Strength"},
    color_discrete_map=color_map
)

# Fix y-axis to show true scale
fig.update_layout(yaxis_range=[0, df["Activation"].max() * 1.1])

filename = "experiment_4_superposition_179.html"
fig.write_html(filename)
print(f"\nSUCCESS. Graph saved to {filename}")
print("Check the HTML: Do you see the Gold bars for 'Web' and 'Cloud'?")