Markdown

# The Asymmetry of Intent

**Structural Alignment in Large Language Models**

This repository contains the experimental code, and visualizations for the paper *"The Asymmetry of Intent."*

This project explores the geometric collision between biological cognition (Convergent) and computational cognition (Divergent). Using mechanistic interpretability techniques, we measure the "Thermodynamics of Intent"—quantifying the entropy gap when a model attempts to force divergent probability flows into convergent semantic funnels.

## 📂 Repository Structure

- `scripts/` - Python scripts for running the experiments.
- `visualizations/` - Generated plots used in the paper.

## 🧪 The Experiments

### 1. The Thermodynamics of Intent (`experiment_1_thermodynamics.py`)
Utilizes the **Logit Lens** to track the model's internal uncertainty (entropy) layer-by-layer when processing "Convergent" vs. "Divergent" prompts.
* **Key Finding:** The model acts as a "narrow funnel" for convergent constraints (Entropy: **0.67**), but refuses to collapse for divergent constraints (Entropy: **1.28**). This **0.61 thermodynamic gap** quantifies the "Geometric Frustration" of trying to force a probability engine to act as a stop sign.

### 2. The Gravity Model (`experiment_2_gravity.py`)
Calculates the "Center of Mass" for factual vs. hallucinated statements to measure the geometric tension of Truth.
* **Key Finding:** Factual statements maintain high positive tension on the "Truth Ridge"; hallucinations fall into the "Lie Valley."

### 3. Asymmetry Mending (`experiment_3_mending.py`)
Demonstrates the "Recoil" mechanics of correcting a model's trajectory. We forced the model to hallucinate ("The dog is a type of reptile...") and measured the force required to snap it back to reality.
* **Key Test:** Comparing natural generation drift vs. steered generation using a "Truth Vector" (First-Order Constraint).
* **Key Finding:** The "Cyan Line" (Mended) physically tears away from the "Red Line" (Drift), proving that alignment requires a geometric "Recoil" rather than just a nudge.

### 4. Superposition & Geometric Frustration (`experiment_4_superposition.py`)
Identifies **Neuron 179**, a polysemantic neuron that activates for both "Technology" and "Nature."
* **Key Finding:** Ambiguous words like "Web" and "Cloud" trigger the neuron at equal intensity, proving that multiple meanings exist in superposition until collapsed by context.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch

### Dependencies
Install the required libraries using Python's package manager:


python -m pip install torch numpy pandas plotly tqdm transformer_lens sentence-transformers transformers

### Running the Code
To reproduce the entropy gap analysis and generate the graphs:

bash
python scripts/experiment_1_thermodynamics.py --model gpt2-small


🧰 Tools & Technologies

This project relies on the following libraries for mechanistic interpretability and vector analysis:

    TransformerLens: Used for hooking into internal model activations (specifically utilizing the Logit Lens technique), caching attention patterns, and patching activations to test "Ego Death" states.

    Sentence Transformers (SBERT): Used for calculating vector similarity and measuring the "Drift" of concepts across layers.

    Hugging Face Transformers: Provides the base GPT-2 Small model architecture.

    Plotly: Used for interactive 3D visualizations of the vector landscapes.

📚 Citations & Attribution

If you use this code or the concepts from the paper, please cite the original essay and the following tools that made this analysis possible:
Framework & Analysis

    North, J. R. (2025). The Asymmetry of Intent: Structural Alignment in Large Language Models. [Repository Link]

Libraries Used

TransformerLens Nanda, N., & Bloom, J. (2022). TransformerLens. A library for mechanistic interpretability of GPT-2 style language models.
Code snippet

@misc{nanda2022transformerlens,
    title = {TransformerLens},
    author = {Neel Nanda and Joseph Bloom},
    year = {2022},
    howpublished = {\url{[https://github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)}},
}

Sentence Transformers Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
Code snippet

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    year = "2019",
}

GPT-2 (Model Architecture) Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
📜 License

This project is licensed under the MIT License - see the LICENSE file for details.