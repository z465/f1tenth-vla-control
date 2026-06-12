import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# 1. process command line inputs

parser = argparse.ArgumentParser()

parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--question", type=str, default="What animal is in this image?")
parser.add_argument("--max_new_tokens", type=int, default=64)

# 1.2 reading arguments from command line and put them into args
args = parser.parse_args()

# 2. Load foundation model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.base_model,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

# 3. Preprocessing image inputs
processor = AutoProcessor.from_pretrained(args.base_model)

# 3.1. Switch to evaluation mode
model.eval()    # Entering Inference mode

# 3.2 Constructing messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": args.image_path},
            {"type": "text", "text": args.question},
        ],
    }
]

# 3.3 Apply chat template
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True, # add at last at the position of the assistant begins answering the question, otherwise the model doesn't know to continue generating answer of assistant
)

# 3.4 Process image inputs
image_inputs, video_inputs = process_vision_info(messages)

# 4. Processor turning inputs into tensors
inputs = processor(
    text=[text],    #[], because processor allows batch input
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# 4.1 move inputs to the same device of the model
inputs = inputs.to(model.device)

# 5. Generate answers
with torch.no_grad(): 
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,    # do_sample=False: meaning no random sampling, generate deterministic answer
    )

# 5.1 Removing inputs part, reserving only the newly generated answers
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 5.2 Decode into normal text
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

# 5.3 Print out the answers
print("\nQuestion:")
print(args.question)
print("\nAnswer:")
print(output_text[0])

