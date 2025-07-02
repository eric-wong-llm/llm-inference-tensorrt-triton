import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Need to set pad token because GPT-2 does not have pad_token, so that attention_mask can correctly tell the model which tokens are padding
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval() # Switch model from training mode to evaluation (inference) mode

# Wrapper to avoid cache and simplify export
class GPT2Wrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask):
        # Only outputs logits, no caching
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits

wrapper_model = GPT2Wrapper(model)

# Tokenize a sample input
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt", padding=True) # Inputs are padded to the same length in a batch

# Ensure attention_mask is provided (some models require it)
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

# Export the model to ONNX
torch.onnx.export(
    wrapper_model,
    args=(input_ids, attention_mask),
    f="onnx/gpt2.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size", 1: "seq_length"}
    },
    do_constant_folding=True,
    opset_version=14
)

print("✅ Exported GPT-2 model to onnx/gpt2.onnx")
