# F1TENTH VLA Control

This repository is based on the original [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) project.

I extended the original project with a F1TENTH VLM/VLA control pipeline for the AutoDRIVE Simulator and AutoDRIVE DevKit. The goal is to use front-camera images from the simulated F1TENTH vehicle as input to a vision-language model, generate a short reasoning result, and output a driving action suggestion.

## My Additions

This repository adds the following components:

* `f1tenth_vlm_infer.py`:
  Uses images captured from the F1TENTH front camera as input to the VLM.
  The script outputs:

  * a short reasoning result explaining why the action was selected
  * the final driving action suggestion

The supported action outputs are:

* `forward`
* `turn left`
* `turn right`
* `stop`
* `backward`

## Original Project

The original project provides Qwen-VL fine-tuning and inference utilities.
This repository keeps the original functionality and adds the F1TENTH control-related inference pipeline on top of it.

## Usage

Run `f1tenth_vlm_infer.py` with an image directory:

Example:

```bash
python scripts/f1tenth_vlm_infer.py \
  --image_dir /workspace/Qwen-VL-Series-Finetune/data/images \
  --max_images 600
```

## Optional Command-Line Arguments

The script supports the following optional arguments:

```bash
--base_model str
--image_dir str
--max_images int
--question str
--max_new_tokens int
```

## Shared Folder Setup

This repository is designed to work together with a separate ROS2 control repository, such as `f1tenth-ros2-control`.

The two repositories communicate through a shared data folder.

Expected image folder:

```text
.../Qwen-VL-Series-Finetune/data/images
```

Typical shared folder structure:

```text
Qwen-VL-Series-Finetune/
└── data/
    ├── images/
    │   ├── camera_0001.jpg
    │   ├── camera_0002.jpg
    │   └── ...
    ├── actions.txt
    └── latest_action.txt
```

The ROS2 control repository saves front-camera images into:

```text
data/images/
```

This repository reads the latest image from that folder, performs VLM inference, and writes the selected action to:

```text
latest_action.txt
```

It can also save the full inference history, including reasoning and action results, to:

```text
actions.txt
```

## Relation to `f1tenth-ros2-control`

This repository is responsible for:

* loading the VLM
* reading front-camera images
* generating reasoning and action suggestions
* writing the latest action to a shared file

The ROS2 control repository is responsible for:

* subscribing to the F1TENTH front camera topic
* saving images to the shared folder
* reading the latest action
* publishing throttle and steering commands to AutoDRIVE DevKit

