import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import time
from pathlib import Path
import os


# utilities functions
def extract_action(raw_answer):
    text = raw_answer.lower()

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("action"):
            action = line.replace("action:", "").strip()
            return action



actions = []

# 1. Processing the command line arguments and store them into args

parser = argparse.ArgumentParser()

parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
## add the image directory to avoid execute once again this py file everytime you give an image as an input
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--max_images", type= int, default=10)

parser.add_argument("--question", type=str, default="Identify the drivable corridor direction and choose the safest action.")
parser.add_argument("--max_new_tokens", type=int, default=64)

args = parser.parse_args()

# 2. Loading foundation model
# measuring the time of inference (test)
print("Loading base model...")


model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.base_model,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

# 3. Preprocessing images and text inputs
# 3.1 Load processor
processor = AutoProcessor.from_pretrained(args.base_model)

# 3.2 text perspective:
## switch to evaluation mode
model.eval()

# Construct messages

SYSTEM_CONTEXT = """

You are an autonomous driving assistant for a simulated F1TENTH car.

The input image is a single frame from the car's front camera.
Choose one safe short-term driving action.

Use the car's perspective:
- image left = car left
- image right = car right

Allowed actions:
forward
turn left
turn right
stop
backward

Scene understanding:
- This is a closed driving track.
- The gray ground is the drivable floor.
- The black wall, dark boundary, black barrier, or large dark object is not drivable.
- A wall across the front view usually does not mean a dead end.
- If a wall appears across the front in the distance, the track probably turns left or right.
- In that case, look for where the open gray drivable floor continues and choose a turn.

Wall and collision rules:
- Do not choose "backward" just because a wall is visible ahead in the distance.
- Choose "backward" only if the car has already hit the wall, is stuck, or the front tires are touching or almost touching the wall.
- If the car is very close to a wall or obstacle, choose "backward" first.
- If there is still open gray drivable floor between the car and the wall, do not choose "backward"; choose a turn instead.
- Do not choose "forward" if a wall blocks the forward path.

Right-turn visual pattern:
- If the open gray drivable floor continues or widens toward the right side of the image, choose "turn right".
- If a wall is ahead but the right side has more open gray floor, choose "turn right".
- If the left side or front-left side is mostly wall and the right side is open gray floor, choose "turn right".

Left-turn visual pattern:
- If the open gray drivable floor continues or widens toward the left side of the image, choose "turn left".
- If a wall is ahead but the left side has more open gray floor, choose "turn left".
- If the right side or front-right side is mostly wall and the left side is open gray floor, choose "turn left".

Action rules:
- Choose "forward" only if the open gray drivable floor continues mostly straight ahead and is clear.
- Choose "turn right" if the track continues to the right or the open gray floor is mainly on the right.
- Choose "turn left" if the track continues to the left or the open gray floor is mainly on the left.
- Choose "backward" only when the car is touching, almost touching, or stuck against a wall or obstacle.
- Choose "stop" only if no safe drivable floor is visible.

Common mistake to avoid:
- Do not treat a distant wall as a dead end.
- On this track, a wall across the front usually means the car should turn, not reverse.
- First try to follow the open gray drivable floor.
- Reverse only after collision or when the car is already too close to the wall.

Output format:
reason: one short sentence based on the open gray floor, wall distance, and track direction
action: exactly one of [forward, turn left, turn right, stop, backward]

Do not output anything else.

"""



image_path_pre = None
processed_count = 0
image_dir = Path(args.image_dir)

while processed_count < args.max_images:

    # Constructing image file path
    image_paths = (
        list(image_dir.glob("*.png")) + 
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg"))
    )

    if len(image_paths) == 0:
        print("No new image found. Waiting...")
        time.sleep(1)
        continue

    image_paths.sort(key=lambda x: os.path.getmtime(x))   
    latest_image_path = image_paths[-1]

    if latest_image_path == image_path_pre:
        print("No new image. Waiting...")
        time.sleep(1)
        continue
    
    time.sleep(0.1) # wait briefly to reduce the chance of reading a file that is still being written
    processed_count += 1

    # measuring the time of inference (test)
    start_time = time.time()

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_CONTEXT},
            ]

        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(latest_image_path)},
                {"type": "text", "text": args.question},
            ]
        }
    ]

    # apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 3.3 images & videos side
    image_inputs, video_inputs = process_vision_info(messages)

    # 3.4 processor turning inputs into tensors
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )

    # 4. Model generates the answer tokenids
    # 4.1 putting inputs to the same device of the model
    inputs = inputs.to(model.device)

    # 4.2 generating answer
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # 4.3 preserve only the generated outputs
    generated_ids_trimmed = [out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)]

    # 5. Decode the trimmed ids into texts
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    end_time = time.time()
    duration = end_time - start_time

    # 6. Converting the vlm outputs command into ros2 actions
    # 6.1 bridging VLM outputs and ROS control via .txt file

    # cutting the output into reason and action
    raw_output = output_text[0]
    action = extract_action(raw_output)

    with open("/workspace/Qwen-VL-Series-Finetune/data/actions.txt", "a") as file:
        file.write(f"{latest_image_path.name},{raw_output}\n")
    ## Because the ros2 node need to read the latest action so "w"
    with open("/workspace/Qwen-VL-Series-Finetune/data/latest_action.txt", "w") as file:
        file.write(f"{action}\n")



    # 7. Print
    print("=====================================================================")
    print(f"- Image: {latest_image_path}")
    print(f"\n- Question:")
    print(args.question)
    print("---------------------------------------------------------------------")
    print(f"\n- Answer:")
    print(output_text[0])
    print("*********************************************************************")
    print(f"Duration of inference is: {duration}s.")
    print("*********************************************************************")
    print("=====================================================================")
    print("\n")

    image_path_pre = latest_image_path







    



