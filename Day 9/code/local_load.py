from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "gpt2"  # Using GPT-2 as it is very lightweight for testing

# 1. Load the tokenizer (converts text to numbers)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Load the model (the brain)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 3. Prepare the input
input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt")

# 4. Generate output tokens
with torch.no_grad():
    output_tokens = model.generate(**inputs, max_new_tokens=10)

# 5. Decode numbers back into text
result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(result)
