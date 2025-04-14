# as per https://dev.to/thenomadevel/how-to-fine-tune-models-easily-with-peft-2pn7

#import dbl
import transformers
import datasets
import trl
import peft
import torch

# filename = "databricks-dolly-15k.jsonl"
# dataset = dbl.load_file(filename)

hf_dataset = "databricks/databricks-dolly-15k"
dataset = datasets.load_dataset(hf_dataset)

size="3b"
checkpoint_name = f"britllm/britllm-{size}-v0.1"

tokeniser = transformers.AutoTokenizer.from_pretrained(checkpoint_name)

model = transformers.AutoModelForCausalLM.from_pretrained(
    checkpoint_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model, tokeniser = trl.setup_chat_format(model=model, tokenizer=tokeniser)

peft_config = peft.LoraConfig(
    r=6,  # Rank dimension for the update matrices
    lora_alpha=8,  # Scaling factor
    lora_dropout=0.05,  # Dropout rate
    target_modules="all-linear",  # Apply LoRA to all linear layers
    task_type="CAUSAL_LM",  # Specify task type
)

args = trl.SFTConfig(
    output_dir="Peft_wgts",  # Directory to save model checkpoints
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size per device
    gradient_accumulation_steps=2,  # Accumulate gradients for larger batches
    gradient_checkpointing=True,  # Save memory by re-computing gradients
    learning_rate=2e-4,  # Learning rate
    bf16=True,  # Enable mixed precision
)

trainer = trl.SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()