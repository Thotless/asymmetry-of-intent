import torch
import numpy as np
import plotly.graph_objects as go
from transformer_lens import HookedTransformer

# 1. SETUP
print("Initializing the Repair Protocol...")
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

# 2. DEFINE THE TRUTH VECTOR (The "Ridge" Direction)
# We use the Biology axis again.
facts = ["Dogs are mammals", "Trees grow", "Fish swim", "Humans breathe"]
lies = ["Dogs are reptiles", "Trees fly", "Fish walk", "Humans stop breathing"]

def get_center(sentences):
    vecs = []
    for s in sentences:
        _, cache = model.run_with_cache(s)
        vecs.append(cache["resid_post", 10][0, -1, :].cpu().detach().numpy())
    return np.mean(vecs, axis=0)

truth_vector = get_center(facts) - get_center(lies)
truth_vector = truth_vector / np.linalg.norm(truth_vector) # Normalize length

# 3. DEFINE THE REPAIR HOOK
# This forces the model's internal state to move toward the Truth Ridge.
def steering_hook(resid_pre, hook):
    coeff = 15.0 # High strength to force the "Snap"
    return resid_pre + (torch.tensor(truth_vector).float().to(resid_pre.device) * coeff)

# 4. THE EXPERIMENT
# We will generate TWO versions of the future from the same bad start.
bad_prompt = "The dog is a type of reptile which"

# FUNCTION: Generate and Record Tension
def generate_track(prompt, use_hook=False):
    input_ids = model.to_tokens(prompt)
    scores = []
    words = []
    
    # If we are steering, we prepare the hook context
    hooks = [("blocks.6.hook_resid_pre", steering_hook)] if use_hook else []
    
    # We use a loop to generate AND measure step-by-step
    for _ in range(15):
        # We need to re-apply the hook every step if enabled
        with model.hooks(fwd_hooks=hooks):
            logits, cache = model.run_with_cache(input_ids)
        
        # MEASURE TENSION (Layer 10)
        current_vec = cache["resid_post", 10][0, -1, :].cpu().detach().numpy()
        tension = np.dot(current_vec, truth_vector)
        scores.append(tension)
        
        # PREDICT
        next_token = logits[0, -1].argmax()
        word = model.to_string(next_token)
        words.append(word)
        input_ids = torch.cat([input_ids, next_token.reshape(1, 1)], dim=1)
        
    return words, scores

print("\n1. Generating Control (Unsupervised Hallucination)...")
words_control, scores_control = generate_track(bad_prompt, use_hook=False)
sent_control = bad_prompt + "".join(words_control)
print(f"CONTROL OUT: '{sent_control}'")

print("\n2. Generating Mended (Steered towards Truth)...")
words_mended, scores_mended = generate_track(bad_prompt, use_hook=True)
sent_mended = bad_prompt + "".join(words_mended)
print(f"MENDED OUT: '{sent_mended}'")

# 5. VISUALIZE THE SNAP
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=scores_control, x=words_control,
    mode='lines+markers', name='Natural Drift (The Lie)',
    line=dict(color='red', dash='dot')
))

fig.add_trace(go.Scatter(
    y=scores_mended, x=words_mended,
    mode='lines+markers', name='Mended Trajectory',
    line=dict(color='cyan', width=4)
))

fig.update_layout(
    title="Asymmetry Mending: Force-Correcting a Hallucination",
    xaxis_title="Generated Future",
    yaxis_title="Truth Tension",
    template="plotly_dark"
)

fig.write_html("experiment_3_mending.html")
print("\nSuccess! Open 'experiment_3_mending.html' to see the geometric correction.")