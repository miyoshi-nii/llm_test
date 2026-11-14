# LLM Test
A study repository for large language models: toy notebooks and notes, starting from the Transformer architecture.

## Papers with Codes

### 1. Foundations: The Transformer

How self-attention, multi-head attention and positional encoding, stacked into an encoder–decoder, replace recurrence and convolution entirely (Transformer).

- **Transformer**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.
  "Attention Is All You Need." NeurIPS 2017.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) / Official code: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) / Notebook: [docs/transformer.ipynb](docs/transformer.ipynb)

### 2. Decoder-only Language Models and Scaling Laws

How a decoder-only Transformer trained only on next-token prediction becomes a general-purpose language model (GPT-2), and how its loss scales with parameters, data and compute (scaling laws, Chinchilla).

- **GPT-2**: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever.
  "Language Models are Unsupervised Multitask Learners." OpenAI Technical Report, 2019.
  [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) / Official code: [openai/gpt-2](https://github.com/openai/gpt-2) / Notebook: [docs/gpt2.ipynb](docs/gpt2.ipynb)
- **Scaling Laws**: Jared Kaplan, Sam McCandlish, Tom Henighan, et al.
  "Scaling Laws for Neural Language Models." arXiv 2020.
  [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) / Notebook: [docs/gpt2.ipynb](docs/gpt2.ipynb) (Section 11)
- **Chinchilla**: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al.
  "Training Compute-Optimal Large Language Models." NeurIPS 2022.
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) / Notebook: [docs/gpt2.ipynb](docs/gpt2.ipynb) (Section 11)

### 3. Parameter-Efficient Fine-Tuning: LoRA

Adapting a pretrained model by freezing its weights and training only a low-rank update ΔW = BA, which cuts the trainable parameters by orders of magnitude and can be merged back into the weights at no inference cost (LoRA).

- **LoRA**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.
  "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) / Official code: [microsoft/LoRA](https://github.com/microsoft/LoRA) / Notebook: [docs/lora.ipynb](docs/lora.ipynb)

### 4. Learning from Human Preferences: DPO and PPO (RLHF)

Aligning a language model with human preferences, either by reinforcement learning against a reward model with a KL penalty to a reference policy (RLHF with PPO), or directly from preference pairs with a classification-style loss whose optimum is the same KL-regularized policy (DPO).

- **DPO**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn.
  "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) / Official code: [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) / Notebook: [docs/dpo.ipynb](docs/dpo.ipynb)
- **PPO**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
  "Proximal Policy Optimization Algorithms." arXiv 2017.
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) / Official code: [openai/baselines](https://github.com/openai/baselines) / Notebook: [docs/ppo.ipynb](docs/ppo.ipynb)
- **InstructGPT (RLHF)**: Long Ouyang, Jeff Wu, Xu Jiang, et al.
  "Training language models to follow instructions with human feedback." NeurIPS 2022.
  [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) / Notebook: [docs/ppo.ipynb](docs/ppo.ipynb) (Section 6)

## Notebooks

All notebooks live in [docs/](docs/). Read them in this order:

| Notebook | What it covers | Runs on |
|---|---|---|
| [transformer.ipynb](docs/transformer.ipynb) | A from-scratch NumPy forward pass of the encoder–decoder Transformer with random weights (no training), checking each equation numerically: embeddings and sinusoidal positional encoding, scaled dot-product attention and why it divides by √d_k, causal and padding masks, multi-head attention, permutation equivariance, LayerNorm (Post-LN vs. Pre-LN), the FFN, cross-attention, the weight-tied output layer, greedy decoding with a KV cache, and parameter and FLOP counts | CPU (Colab) |
| [gpt2.ipynb](docs/gpt2.ipynb) | GPT-2 small (124M) implemented from scratch in NumPy and loaded with the official pretrained weights, verified against `transformers`: byte-level BPE, learned position embeddings, Pre-LN blocks, next-token prediction, attention sinks, previous-token and induction heads, the logit lens, massive activations, KV-cache decoding with greedy / temperature / top-k / top-p sampling, and scaling laws (C ≈ 6ND checked by counting FLOPs, Kaplan's power laws, Chinchilla's compute-optimal allocation) | CPU (Colab) |
| [lora.ipynb](docs/lora.ipynb) | LoRA from the equations up: parameter count, zero-initialized B, rank ≤ r, merging, and gradients as projections of the full fine-tuning gradient (checked numerically); a toy regression where the loss reached at each rank matches the Eckart–Young bound; then a hand-written LoRA layer on GPT-2 small trained for a few dozen steps on fictional facts, checking frozen base weights, merge / unmerge, switching the adapter on and off, a 1 MB adapter file, agreement with `peft`, and learning vs. forgetting compared with full fine-tuning | CPU (Colab; a GPU makes Section 4 faster) |
| [dpo.ipynb](docs/dpo.ipynb) | DPO from the equations up: the Bradley–Terry model, the optimal policy of KL-regularized reward maximization, how the partition function cancels to give the DPO loss, and its gradient (checked numerically); a toy problem where DPO converges exactly to the optimal policy and, on finite data, matches "reward model + KL-regularized RL"; then GPT-2 small with a LoRA policy (the reference policy is the same model with the adapter off) trained on movie-review preferences, showing the loss starting at log 2, the shift to positive continuations, and over-optimization when training continues | CPU (Colab; a GPU makes Section 6 faster) |
| [ppo.ipynb](docs/ppo.ipynb) | PPO from the equations up: REINFORCE and baselines, GAE, the importance-sampled surrogate objective checked against exact dynamic programming (policy gradient theorem, performance difference lemma, surrogate error growing with KL), the clipped objective and KL estimators; PG vs. unclipped vs. clipped PPO on a small MDP; then RLHF-style PPO on GPT-2 small (LoRA policy, value head, per-token KL penalty, final-token reward), ending in reward hacking of a word-list reward | GPU recommended for Section 6 (Colab; runs on CPU but slowly) |
