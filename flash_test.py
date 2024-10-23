import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler
from peft import PeftModel

# Load LoRA
transformer = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="transformer",
    torch_dtype=torch.float32,
)
transformer = PeftModel.from_pretrained(transformer, "jasperai/flash-sd3")


# Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    transformer=transformer,
    torch_dtype=torch.float32,
    text_encoder_3=None,
    tokenizer_3=None
)

# Scheduler
pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
  "stabilityai/stable-diffusion-3-medium-diffusers",
  subfolder="scheduler",
)

pipe.to("cpu")

prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."

image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
image.save("output_image.png")