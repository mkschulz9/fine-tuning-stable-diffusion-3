import os, torch
from diffusers import DiffusionPipeline
from datasets import Dataset

class ModelProcessor:
    def __init__(self, model_id: str):
        """Load model from HuggingFace"""
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.float16,  # Mixed precision (faster inference)
            #force_download=True
        ).to("cuda")  # Move model to GPU (faster processing)
        
        # self.pipe.enable_attention_slicing() # Enable if we need to save some memory in exchange for small speed decrease (https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/image_variation#diffusers.StableDiffusionImageVariationPipeline.enable_attention_slicing)
        # tips if running out of gpu memory: https://huggingface.co/learn/diffusion-course/en/unit3/2#generating-images-from-text
        
        print(f"\nModel loaded: {model_id}")
        
    def generate_images(self, test_dataset: Dataset, num_images: int = None, output_dir: str = "generated_images"):
      """Generate images using model's pipeline on prompts in test dataset"""
      pass