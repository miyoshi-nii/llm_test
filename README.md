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

## Notebooks

All notebooks live in [docs/](docs/). Read them in this order:

| Notebook | What it covers | Runs on |
|---|---|---|
| [transformer.ipynb](docs/transformer.ipynb) | A from-scratch NumPy forward pass of the encoder–decoder Transformer with random weights (no training), checking each equation numerically: embeddings and sinusoidal positional encoding, scaled dot-product attention and why it divides by √d_k, causal and padding masks, multi-head attention, permutation equivariance, LayerNorm (Post-LN vs. Pre-LN), the FFN, cross-attention, the weight-tied output layer, greedy decoding with a KV cache, and parameter and FLOP counts | CPU (Colab) |
| [gpt2.ipynb](docs/gpt2.ipynb) | GPT-2 small (124M) implemented from scratch in NumPy and loaded with the official pretrained weights, verified against `transformers`: byte-level BPE, learned position embeddings, Pre-LN blocks, next-token prediction, attention sinks, previous-token and induction heads, the logit lens, massive activations, KV-cache decoding with greedy / temperature / top-k / top-p sampling, and scaling laws (C ≈ 6ND checked by counting FLOPs, Kaplan's power laws, Chinchilla's compute-optimal allocation) | CPU (Colab) |
