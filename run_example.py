from PIL import Image
import numpy as np
import argparse
import os
from omegaconf import OmegaConf
import mindspore as ms
from pilot_ms.pipeline.pipeline_pilot import PilotPipeline


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", type = str, default= "coco1.yaml"
)

args = parser.parse_args()
config = OmegaConf.load(args.config_file)

if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

prompt_list = [config.prompt]
controlnet = None
adapter = None

model_list = ["base"]

if config.fp16:
    weight_format = ms.float16
else:
    weight_format = ms.float32

image = Image.open(config.input_image).convert("RGB")
image = image.resize((config.W, config.H), Image.NEAREST)
mask_image = Image.open(config.mask_image).convert("RGB")
mask_image = mask_image.resize((config.W, config.H), Image.NEAREST)
if mask_image.mode != "RGB":
    mask_image = mask_image.convert("RGB")
for x in range(config.W):
    for y in range(config.H):
        r, g, b = mask_image.getpixel((x, y))
        if (r, g, b) != (0, 0, 0) and (r, g, b) != (255, 255, 255):
            mask_image.putpixel((x, y), (0, 0, 0))

################################### loading models and additional controls #############################
#TODO:正在迁移：

# load base model
print("load base model")
if config.fp16:
    pipe = PilotPipeline.from_pretrained(
        f"{config.model_path}/{config.model_id}",
        controlnet = controlnet,
        adapter = adapter,
        torch_dtype = ms.float16,
        variant = "fp16",
        requires_safety_checker = False,
    )
else:
    pipe = PilotPipeline.from_pretrained(
        f"{config.model_path}/{config.model_id}",
        controlnet = controlnet,
        adapter = adapter,
        torch_dtype = ms.float16,
        requires_safety_checker = False,
    )
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
generator = ms.Generator().manual_seed(config.seed)
pipe.to("cuda", weight_format)

#################################### run examples and save results ##########################

