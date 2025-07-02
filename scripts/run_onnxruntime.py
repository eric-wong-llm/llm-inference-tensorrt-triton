import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer (GPT-2)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Sample input text
input_text = "The future of AI is"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="np")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Load ONNX model
session = ort.InferenceSession("onnx/gpt2.onnx")

# Run inference
outputs = session.run(
    None,
    {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attention_mask.astype(np.int64)
    }
)

# Get logits
logits = outputs[0]

# Get next token (greedy decoding)
next_token_id = np.argmax(logits[:, -1, :], axis=-1)

# Append token and decode
new_ids = np.concatenate([input_ids, next_token_id[:, None]], axis=-1)
output_text = tokenizer.decode(new_ids[0])

print("🧠 Generated:", output_text)

