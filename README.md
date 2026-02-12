# chess_rl

Fine-tuning a large language model to play chess using reinforcement learning. The project trains a **Qwen2.5** model through a two-stage pipeline: supervised fine-tuning (SFT) on legal chess moves, followed by RL training (GRPO) to improve move quality and game outcomes.

---

## How it works

### Stage 1 — Supervised Fine-Tuning (SFT)

The model is first trained on chess positions and correct moves in standard algebraic notation (SAN). This gives it a baseline understanding of legal move generation before RL training begins.

- `generate_answers.ipynb` — generates training examples from chess positions
- `sft_trainng.ipynb` — initial SFT run
- `sft_training_better.ipynb` — improved SFT with refined data/prompting

### Stage 2 — Reinforcement Learning (GRPO)

After SFT, the model is further trained using Group Relative Policy Optimization (GRPO). The reward signal penalizes illegal moves and rewards moves that improve position evaluation (e.g. via Stockfish scoring).

Configuration for both stages lives in `config.yaml`.

---

## Project structure

```
chess_rl/
├── generate_answers.ipynb      # Generate SFT training data
├── sft_trainng.ipynb           # SFT training (v1)
├── sft_training_better.ipynb   # SFT training (v2, improved)
├── test.py                     # Evaluate a trained model
├── config.yaml                 # Training hyperparameters and paths
└── .env.example.txt            # Environment variable template
```

---

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/czovekboti/chess_rl.git
cd chess_rl
pip install torch transformers trl peft chess python-dotenv pyyaml
```

Copy the environment template and fill in your values:

```bash
cp .env.example.txt .env
```

---

## Configuration

Edit `config.yaml` to set model paths, dataset paths, and training hyperparameters before running any notebook or script.

---

## Usage

### 1. Generate training data

Open and run `generate_answers.ipynb` to produce the SFT dataset.

### 2. Run SFT

Run `sft_training_better.ipynb` to fine-tune the base Qwen2.5 model on the generated data.

### 3. Run RL training

RL training is configured via `config.yaml`. Launch after SFT completes.

### 4. Test a trained model

```bash
python test.py <model_name>
```

`<model_name>` should be the path or HuggingFace identifier of your fine-tuned model checkpoint.

---

## Results

> _Add training curves, example games, or Stockfish evaluation scores here._

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) / [TRL](https://github.com/huggingface/trl) for efficient fine-tuning
- [python-chess](https://github.com/niklasf/python-chess) for board representation and move validation
- [Stockfish](https://stockfishchess.org/) for position evaluation
