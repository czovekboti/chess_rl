import os
from dotenv import load_dotenv
load_dotenv()
stockfish_path = os.getenv("STOCKFISH_PATH")
import argparse
parser = argparse.ArgumentParser(description="Choose model configuration")
parser.add_argument(
    "config_name",                  # pozicionális argumentum
    type=str,
    help="Available models: llama, phi, mistral"
)
import yaml
path = 'config.yaml'
def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(path)
args = parser.parse_args()
config_name = args.config_name
print(config_name)
match config_name:
    case "llama":
        config = config["llama_config"]
    case "phi":
        config = config["PHI_config"]
    case "mistral":
        config = config["mistral_config"]
    case "qwen7b":
        config = config["qwen7b_config"]
    case "qwen4b":
        config = config["qwen4b_config"]
    case _:
        print("Check model name – perhaps the keyboard got excited.")

        



from unsloth import FastLanguageModel
import torch
max_seq_length = config['max_seq_length']# Can increase for longer reasoning traces
lora_rank = config['lora_rank'] # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config["model"],
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
from datasets import load_dataset
dataset = load_dataset("czovekboti/chessdata", split="train")
# Initialize wandb
import wandb
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_PROJECT"] = "Chess_RL_Project"
os.environ["WANDB_ENTITY"] = "czovekboti-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem"
wandb.login()
wandb.init(
    project="Chess_RL_Project",
    entity = "",
    name=config["name"],
    config={
        "model": config["model"],
        "max_seq_length": config['max_seq_length'],
        "lora_rank": lora_rank,
        "learning_rate": config["learning_rate"],
        "max_steps": config["max_steps"],
    }
)
SYSTEM_PROMPT = """
You are a chess coach assistant. You will be given a board position in FEN format. Your job is to analyze the board and suggest the best legal move for the player whose turn it is.

Please follow this exact format in your response:

<reasoning>
(Brief explanation of what you see on the board — piece activity, threats, and candidate moves)
</reasoning>
<answer>
(best move written in correct SAN format, such as Nf3 or exd5)
</answer>

Do not invent illegal or impossible moves. The move must be legal in the given FEN position.
Do not use UCI format like e2e4 — only SAN notation like e4, Nf3, or O-O.
In case of taking a piece use the [file]x[target square] format
### Example:
FEN: rnbqkbnr/pppppppp/8/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 1

<reasoning>
White has just played e4 and developed the knight to f3. It’s Black’s turn. The e4 pawn is undefended. Capturing it with the pawn from d7 to d5 is a natural central counter.
</reasoning>
<answer>
d5
</answer>

Now solve the following position:


"""

# import chess libaries and load engine
import chess, chess.engine
from chess import InvalidMoveError, IllegalMoveError, AmbiguousMoveError
import math
import re
from datasets import load_dataset, Dataset
# Load and prep dataset


XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
def get_board(split = "train"):
    def fen_color(fen: str) -> str:
        return "White" if fen.split()[1] == 'w' else "Black"
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['FEN'] + " You are with the following pieces: " + fen_color(x['FEN'])}
        ], 'evaluation': x['Evaluation'], 'fen': x['FEN']
    }, remove_columns=data.column_names)
    print(data[0])
    return data #


dataset = get_board()
def reward_move(board, dataeval):
  with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    result = engine.analyse(board, chess.engine.Limit(time=1.0)) # time doesn't make a real difference above this
  evaluation = result['score'].relative.score() #evaluation from opponents point of view
  print(f"\n----------------------\n")
  if evaluation is not None:
      scaled_evaluation = math.tanh(evaluation / 900.0) * 2.0 # biggest eval for position in file is around 15000 but 2000+ evals are rare
      if -evaluation > dataeval: # give reward if it improved position (-evaluation cause we need other players pov)
        scaled_evaluation -= 0.5 # -0.5 because the sign is going to be flipped
        print(f"Eval = {-evaluation}, Dataeval = {dataeval}. State was improved->reward = 0.5")
      print(f"Scaled Evaluation: {-scaled_evaluation} ")
      return -scaled_evaluation # *-1 because we need the score of the player who is not in turn
  else:
    return 0.0


# Reward functions
def correctness_reward_func(prompts,fen, completions, evaluation,**kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_moves =  [extract_xml_answer(r) for r in responses]
    fen_str = fen[0] if isinstance(fen, list) else fen
    board = chess.Board(fen_str)
    print(f"------------\nFEN: {fen}\n--------- \nResponse: {responses[0]} \n----------\nExtracted_Move: {extracted_moves[0]}")
    rewards = []
    try:
        if isinstance(evaluation, list):
            evaluation = float(evaluation[0]) # evaluation maybe a list due to a bug
    except (ValueError, TypeError) as e:
        print(f"Error: Could not convert evaluation '{evaluation}' to float. Using default value 0.0.")
        evaluation = 0.0
    # This also checks if the move is right both syntactically and legally
    for move in extracted_moves:
        try:
          board.push_san(move)
          scaled_evaluation = reward_move(board,evaluation) #evaluate board after the move was made
          rewards.append(3.0+scaled_evaluation) # +5
        except InvalidMoveError:
            print(f"\n----------------------\n-1.0 reward for illegal syntax")
            rewards.append(-1.0)
        except ValueError:
            print(f"\n----------------------\n -0.7 reward for illegal move")
            rewards.append(-0.7)
        except AmbiguousMoveError: #meaning two pieces could go to the declared square
            print(f"\n----------------------\n 0.5 reward for right syntax but ambigous move")
            rewards.append(0.5)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\s*.+?\s*</reasoning>\s*<answer>\s*.+?\s*</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r,re.DOTALL) for r in responses]
    return [0.2 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.2 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = float(config["learning_rate"]),
    adam_beta1 = config["adam_beta1"],
    adam_beta2 = config["adam_beta2"],
    weight_decay = config["weight_decay"],
    warmup_ratio = config["warmup_ratio"],
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = config["per_device_train_batch_size"], #2 for bigger model 4 for smaller #16 gb gpu could do 8 with 14b model
    gradient_accumulation_steps = config["gradient_accumulation_steps"], # overall batch size should be 16 or 32 -> sslows training down
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = config["max_steps"],
    save_steps = config["save_steps"],
    max_grad_norm = 0.1,
    report_to = "wandb", # report to weights and biases
    output_dir = "outputs",
    run_name = "chess_llama_grpo",
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()