import torch
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformer
from torch.distributions import Categorical

# 1. SETUP
print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

# 2. DEFINE THE CANDIDATES
# We test 4 different categories to find the one where GPT-2 has the most "Spray"
candidates = [
    {
        "Name": "The Coin Flip",
        "Conv": "The two sides of a coin are Heads and",
        "Div": "I flipped a coin and it landed on"
    },
    {
        "Name": "The Card Suit",
        "Conv": "Hearts, Diamonds, Clubs, and",
        "Div": "I picked a card from the deck. It was the Queen of"
    },
    {
        "Name": "The Menu",
        "Conv": "Peanut butter and",
        "Div": "I ordered a pizza with extra"
    },
    {
        "Name": "The Compass",
        "Conv": "The sun always sets in the",
        "Div": "The wind was blowing from the"
    }
]

# 3. RUN THE TOURNAMENT
print(f"\n{'Candidate':<15} | {'Conv Entropy':<12} | {'Div Entropy':<12} | {'Gap'}")
print("-" * 65)

best_candidate = None
max_gap = -1.0
best_data = []

for cand in candidates:
    # Measure Convergent
    _, cache_c = model.run_with_cache(cand["Conv"])
    logits_c = cache_c["resid_post", -1][0, -1, :] @ model.W_U
    entropy_c = Categorical(logits=logits_c).entropy().item()
    
    # Measure Divergent
    _, cache_d = model.run_with_cache(cand["Div"])
    logits_d = cache_d["resid_post", -1][0, -1, :] @ model.W_U
    entropy_d = Categorical(logits=logits_d).entropy().item()
    
    gap = entropy_d - entropy_c
    
    print(f"{cand['Name']:<15} | {entropy_c:.4f}       | {entropy_d:.4f}       | {gap:.4f}")
    
    # Diagnosis: What was the cliche?
    top_token = model.to_string(logits_d.argmax())
    # print(f"  > Cliche: '{top_token}'") 

    if gap > max_gap:
        max_gap = gap
        best_candidate = cand
        # Save layer-wise data for the winner
        best_data = []
        for layer in range(model.cfg.n_layers):
            # Conv Layer Entropy
            resid_c = cache_c["resid_post", layer][0, -1, :]
            ent_c = Categorical(logits=resid_c @ model.W_U).entropy().item()
            best_data.append({"Layer": layer, "Entropy": ent_c, "Type": "Convergent (The Funnel)"})
            
            # Div Layer Entropy
            resid_d = cache_d["resid_post", layer][0, -1, :]
            ent_d = Categorical(logits=resid_d @ model.W_U).entropy().item()
            best_data.append({"Layer": layer, "Entropy": ent_d, "Type": "Divergent (The Spray)"})

print(f"\nWINNER: {best_candidate['Name']} (Gap: {max_gap:.4f})")
print("Generating graph for the winner...")

# 4. VISUALIZE THE WINNER
df = pd.DataFrame(best_data)

fig = px.line(
    df, x="Layer", y="Entropy", color="Type",
    title=f"The Thermodynamics of Intent: {best_candidate['Name']}",
    labels={"Entropy": "Uncertainty (Shannon)", "Layer": "Model Depth"},
    color_discrete_map={
        "Convergent (The Funnel)": "#00CC96", 
        "Divergent (The Spray)": "#EF553B"
    }
)

# Add annotations
final_conv = df[df["Type"]=="Convergent (The Funnel)"].iloc[-1]["Entropy"]
final_div = df[df["Type"]=="Divergent (The Spray)"].iloc[-1]["Entropy"]

fig.add_annotation(x=11, y=final_conv, text="Collapse", showarrow=True, arrowhead=1, ax=0, ay=-30)
fig.add_annotation(x=11, y=final_div, text="Spray", showarrow=True, arrowhead=1, ax=0, ay=30)

filename = "experiment_1_thermodynamics.html"
fig.write_html(filename)
print(f"SUCCESS. Graph saved to {filename}")