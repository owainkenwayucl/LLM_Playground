from peft import AutoPeftModelForCausalLM

# Load the trained model
model = AutoPeftModelForCausalLM.from_pretrained("./Peft_wgts/checkpoint-2000")

# Merge LoRA and save the full model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./Peft_wgts_merged")
