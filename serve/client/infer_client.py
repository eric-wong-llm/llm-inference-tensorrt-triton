import numpy as np
import tritonclient.http as httpclient
import time
from transformers import AutoTokenizer

# Init client and tokenizer
client = httpclient.InferenceServerClient(url="localhost:8000")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize input
input_text = "The future of AI is"
tokens = tokenizer(input_text, return_tensors="np")
input_ids = tokens["input_ids"].astype(np.int32)
attention_mask = tokens["attention_mask"].astype(np.int32)

# Prepare inputs
inputs = [
    httpclient.InferInput("input_ids", input_ids.shape, "INT32"),
    httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
]
inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(attention_mask)



# Running inference on ONNX model
# Request and measure inference
start = time.time()
response_onnx = client.infer("gpt2_onnx", inputs)
end = time.time()
print(f"⏱️ ONNX inference latency: {end - start:.3f} seconds")

logits_onnx = response_onnx.as_numpy("logits")

# Decode next token
next_token_id_onnx = np.argmax(logits_onnx[:, -1, :], axis=-1)
output_ids_onnx = np.concatenate([input_ids, next_token_id_onnx[:, None]], axis=-1)
output_text_onnx = tokenizer.decode(output_ids_onnx[0])

print("🧠 Generated ONNX Output:", output_text_onnx)
#####################################################################################



# Running inference on TensorRT model
# Request inference
start = time.time()
response_trt = client.infer("gpt2_trt", inputs)
end = time.time()
print(f"⏱️  TensorRT inference latency: {end - start:.3f} seconds"

logits_trt = response_trt.as_numpy("logits")

# Decode next token
next_token_id_trt = np.argmax(logits_trt[:, -1, :], axis=-1)
output_ids_trt = np.concatenate([input_ids, next_token_id_trt[:, None]], axis=-1)
output_text_trt = tokenizer.decode(output_ids_trt[0])

print("🧠 Generated TensorRT Output:", output_text_trt)
#####################################################################################
