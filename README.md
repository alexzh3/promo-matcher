# Promo Matcher – GRPO / LoRA Fine-Tuning (Qwen3 0.6B 4bit)

Concise repo for experimenting with reinforcement-style preference optimization (GRPO / RULER-style comparisons) and LoRA fine‑tuning on the 4‑bit Unsloth Qwen3 0.6B base model, producing lightweight adapters (`promo-matcher-v0`, `promo-matcher-v1`). A local GUI (text-generation-webui) is added for interactive chat and side‑by‑side validation of base vs adapted behavior.

## Repository Layout

Notebooks (experiments):
- [promo_match_colab_v0.ipynb](promo_match_colab_v0.ipynb): Initial GRPO + LoRA training run (model name `promo-matcher-90-v0`).
- [promo_match_colab_v1.ipynb](promo_match_colab_v1.ipynb): Final refined version including comparison / evaluation pass loading base + adapter checkpoints.
- [auto_rl.ipynb](auto_rl.ipynb): Generic task (grammar example) auto RL / RULER style pipeline template.

Adapters / model card stubs:
- [promo-matcher-v0/adapter_config.json](promo-matcher-v0/adapter_config.json)
- [promo-matcher-v1/adapter_config.json](promo-matcher-v1/adapter_config.json)
- Model card placeholder READMEs in each adapter folder.

## Core Components

- Base model: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`
- Parameter-efficient fine-tuning: PEFT LoRA (rank 8, alpha 16, zero dropout) on projection modules (`down_proj`, `up_proj`, `gate_proj`, `q_proj`, `k_proj`, `v_proj`, etc.—see adapter configs).
- Training approach: Iterative generation + preference / comparison style scoring (GRPO / RULER inspired) inside the notebooks.
- Logging (optional): Weights & Biases (W&B) via `WANDB_API_KEY`.
- API usage: Some cells expect `OPENAI_API_KEY` (stub or local compatible endpoint).
- Local inference & evaluation: OpenAI-compatible client objects and direct HF/PEFT model loading for side‑by‑side base vs adapter outputs.

## GUI Chat via text-generation-webui

See: https://github.com/oobabooga/text-generation-webui

Steps:
1. Clone:
   ```
   git clone https://github.com/oobabooga/text-generation-webui.git
   ```
   Recommended to look into the documentation for the full installation. This requires Transformers as back-end to work with the Qwen3-0.6B model.
2. Place the base model under:
   ```
   text-generation-webui/user_data/models/Qwen3-0.6B-unsloth-bnb-4bit
   ```
3. Copy adapter folder(s) (`promo-matcher-v0`, `promo-matcher-v1`) into `text-generation-webui/user_data/loras`.
4. Launch:
   ```
   cd text-generation-webui
   python server.py --auto-devices --chat
   ```
   Or use the .bat files.
5. In the UI:
   - Load the base model.
   - Add/enable LoRA: choose the adapter.
   - Add the system prompt and customise the model parameters.
   - Chat
