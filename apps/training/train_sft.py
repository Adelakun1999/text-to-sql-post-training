import torch
import yaml 
import unsloth

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

from unsloth import FastLanguageModel

#load config
with open("configs/trainings.yml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config['model']['name']
MAX_SEQ_LENGTH = 2048

#Load model + Tokenizer

model , tokenizer = FastLanguageModel.from_pretrained(
    model_name=  MODEL_NAME,
    max_seq_length= MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r = config["lora"]["r"],
    lora_alpha= config["lora"]["alpha"],
    lora_dropout= config["lora"]["dropout"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias = "none",
    use_gradient_checkpointing= "unsloth",
    random_state= 3407
)

#load dataset 
dataset = load_dataset("json", data_files="datasets/sft/train.json", split="train")

#convert message into text 

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize = False
    )

    return {"text" : text}


dataset = dataset.map(format_example)

#Tokenize dataset 

def tokenize_function(example):

    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


dataset = dataset.map(tokenize_function , batched= False)

dataset.set_format(
    type= "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)



trainer = SFTTrainer(
    model = model, 
    processing_class=tokenizer,
    train_dataset= dataset,
    args=TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs = config["training"]["epochs"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        max_steps=config["training"]["max_steps"],
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=3407,
    ),
)

#start training 

trainer.train()

#save model 
model.save_pretrained("models/sql-lora-model")
tokenizer.save_pretrained("models/sql-lora-model")
