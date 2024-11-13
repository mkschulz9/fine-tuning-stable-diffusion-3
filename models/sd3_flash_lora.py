# sd3_flash_lora.py
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

class SD3FlashLora:
    def __init__(self, model_name="stabilityai/stable-diffusion-3-medium-diffusers"):
        """Initialize the model handler without loading the model immediately."""
        self.model_name = model_name
        self.pipe = None  # Placeholder for the pipeline

    def load_base_model(self):
        """Load the base model and initialize the pipeline."""
        transformer = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        
        # Set up the pipeline with the base transformer model
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_name,
            transformer=transformer,
            torch_dtype=torch.float16,
            text_encoder_3=None,
            tokenizer_3=None
        )
        
        # Set the scheduler
        self.pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_name,
            subfolder="scheduler",
        )
        
        # Move pipeline to GPU
        self.pipe.to("cuda")

        for name, module in self.pipe.transformer.named_modules():
          print(name)
    
    def apply_lora(self, lora_rank=16, lora_alpha=32, lora_dropout=0.1):
        """Add LoRA layers to the transformer model."""
        # Ensure the base model is loaded
        if self.pipe is None:
            self.load_base_model()

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["cross_attention", "self_attention"],
            lora_dropout=lora_dropout,
        )

        # Wrap the transformer with LoRA
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)

    def _load_image_from_url(self, url):
        """Load an image from a URL into a PIL format."""
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB")
    
    def fine_tune(self, dataset, epochs=1, lr=5e-4):
        """Fine-tune the model with the provided dataset."""
        # Ensure LoRA layers are applied
        if not isinstance(self.pipe.transformer, PeftModel):
            raise ValueError("LoRA layers have not been applied. Call apply_lora() first.")
        
        # Optimizer only for LoRA parameters
        optimizer = AdamW(self.pipe.transformer.get_peft_parameters(), lr=lr)
        
        self.pipe.transformer.train()  # Set model to training mode
        
        for epoch in range(epochs):
            for sample in dataset[:5]:
                # Retrieve the prompt (caption) and load the image from URL
                prompt = sample['caption']
                image_url = sample['image_url']
                
                # Load target image from URL
                target_image = self._load_image_from_url(image_url)
                
                # Generate the image
                outputs = self.pipe(prompt, num_inference_steps=4, guidance_scale=0)
                generated_image = outputs.images[0]
                
                # Preprocess images to tensors for loss calculation
                target_image = transforms.ToTensor()(target_image).unsqueeze(0).to("cuda")
                generated_image = transforms.ToTensor()(generated_image).unsqueeze(0).to("cuda")
                
                # Calculate loss (e.g., pixel-wise MSE loss for simplicity)
                loss = torch.nn.functional.mse_loss(generated_image, target_image)
                
                # Backpropagation and optimization for LoRA parameters only
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item()}")

    def save_finetuned_model(self, save_path="path_to_save_lora_model"):
        """Save the fine-tuned LoRA parameters."""
        self.pipe.transformer.save_pretrained(save_path)

    def load_finetuned_model(self, lora_path):
        """Load the fine-tuned LoRA model for inference."""
        # Ensure the base model is loaded
        if self.pipe is None:
            self.load_base_model()
        
        # Load the fine-tuned LoRA parameters
        self.pipe.transformer = PeftModel.from_pretrained(self.pipe.transformer, lora_path)

    def run_inference(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        """Run inference on the fine-tuned model with a given prompt."""
        image = self.pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        return image
