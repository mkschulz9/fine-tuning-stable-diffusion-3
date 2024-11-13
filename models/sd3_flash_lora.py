# sd3_flash_lora.py
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image, ImageOps
import requests
from io import BytesIO
from tqdm import tqdm

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
        self.pipe.set_progress_bar_config(disable=True)
        #NAMED MODULES FOR LORA FINETUNING
        #for name, module in self.pipe.transformer.named_modules():
        #  print(name)

    def get_peft_parameters(self):
        """Retrieve LoRA-specific parameters and enable gradient computation."""
        lora_params = []
        for name, param in self.pipe.transformer.named_parameters():
            
            if "lora" in name:
                #print(name)
                param.requires_grad = True  # Enable gradient computation
                lora_params.append(param)
        return lora_params


    def apply_lora(self, lora_rank=16, lora_alpha=32, lora_dropout=0.1):
        """Add LoRA layers to the transformer model."""
        # Ensure the base model is loaded
        if self.pipe is None:
            self.load_base_model()

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules = [
               # First few blocks for initial feature processing
              "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v",
              "transformer_blocks.0.attn.add_q_proj", "transformer_blocks.0.attn.add_k_proj", "transformer_blocks.0.attn.add_v_proj",
              
              # Middle blocks for feature balancing
              "transformer_blocks.4.attn.to_q", "transformer_blocks.4.attn.to_k", "transformer_blocks.4.attn.to_v",
              "transformer_blocks.4.attn.add_q_proj", "transformer_blocks.4.attn.add_k_proj", "transformer_blocks.4.attn.add_v_proj",
              
              # Final blocks for high-level structure
              "transformer_blocks.8.attn.to_q", "transformer_blocks.8.attn.to_k", "transformer_blocks.8.attn.to_v",
              "transformer_blocks.8.attn.add_q_proj", "transformer_blocks.8.attn.add_k_proj", "transformer_blocks.8.attn.add_v_proj"
               ],
            lora_dropout=lora_dropout,
        )

        # Wrap the transformer with LoRA
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
        

        
    def _load_image_from_url(self, url):
        """Load an image from a URL into a PIL format."""
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB")

    def process_target_image(self, target_image):
      # If the target_image is a PIL image, convert it to RGB mode if needed
      if target_image.mode != 'RGB':
          target_image = target_image.convert('RGB')
      
      # Step 1: Calculate padding
      width, height = target_image.size
      if width == height:
          # Already square, just resize
          target_image = target_image.resize((1024, 1024))
      else:
          # Calculate padding to make it square
          delta_w = max(width, height) - width
          delta_h = max(width, height) - height
          padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
          
          # Step 2: Apply padding
          target_image = ImageOps.expand(target_image, padding, fill=(0, 0, 0))  # Padding with black (0, 0, 0)
          
          # Step 3: Resize to (1024, 1024)
          target_image = target_image.resize((1024, 1024))
      return target_image


    def _print_lora_params_state(self, message=""):
            """Prints the state of LoRA parameters, including grad status and initial gradients."""
            print(f"\n{message}")
            for name, param in self.pipe.transformer.named_parameters():
              if "lora" in name:
                print(f"Parameter: {name}")
                print(f" - r.g.: {param.requires_grad}", f" - grad_fn: {param.grad_fn}", f" - C.G.: {param.grad if param.grad is not None else 'No gradient yet'}")
                #print(f" - grad_fn: {param.grad_fn}")
                #print(f" - Current Gradient: {param.grad if param.grad is not None else 'No gradient yet'}\n")

    def fine_tune(self, dataset, epochs=1, lr=5e-4):
        """Fine-tune the model with the provided dataset."""
        # Ensure LoRA layers are applied
        if not isinstance(self.pipe.transformer, PeftModel):
            raise ValueError("LoRA layers have not been applied. Call apply_lora() first.")
        

        
        self.pipe.transformer.train()  # Set model to training mode

        # Optimizer only for LoRA parameters
        lora_params = self.get_peft_parameters()
        #print("LORA PARAMS 1: ", lora_params)
        optimizer = AdamW(lora_params, lr=lr)
        
        #self._print_lora_params_state("LoRA Parameters State Before Training")

        for epoch in range(epochs):
            for idx, sample in enumerate(tqdm(dataset, total=dataset.num_rows, disable=True)):
              if idx >= 1:
                break
    
              prompt = sample['caption']
              image_url = sample['image_url']
              
              # Load target image from URL
              target_image = self._load_image_from_url(image_url)
              target_image = self.process_target_image(target_image)


              #print("LORA PARAMS 2: ", self.get_peft_parameters())

              # Generate the image
              # Generate the image with gradient tracking
              with torch.set_grad_enabled(True):  # Ensures gradients are enabled during generation
                  outputs = self.pipe(prompt, num_inference_steps=4, guidance_scale=0)
                  generated_image = outputs.images[0]

              # After generating, explicitly set requires_grad=True for generated_image tensor
              generated_image = transforms.ToTensor()(generated_image).unsqueeze(0).to("cuda").requires_grad_(True)

              #print("LORA PARAMS 3: ", self.get_peft_parameters())
              # Preprocess images to tensors for loss calculation
              target_image = transforms.ToTensor()(target_image).unsqueeze(0).to("cuda")
              #generated_image = transforms.ToTensor()(generated_image).unsqueeze(0).to("cuda")
              

              #print(f"Generated image requires_grad: {generated_image.requires_grad}")
              #print(f"Target image requires_grad: {target_image.requires_grad}")

              # Calculate loss (e.g., pixel-wise MSE loss for simplicity)
              loss = torch.nn.functional.mse_loss(generated_image, target_image)
              
              #print("LORA PARAMS : ", self.get_peft_parameters())
              # Print the first few values of each LoRA parameter before and after each optimization step
              '''for name, param in self.pipe.transformer.named_parameters():
                  if "lora" in name and param.requires_grad:
                      print(f"{name} - First few values before step: {param.data.flatten()[:5]}")
''' 
              self._print_lora_params_state("LoRA Parameters State Before loss back")

              # Backpropagation and optimization for LoRA parameters only
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              self._print_lora_params_state("LoRA Parameters State AFTER loss back")
              ''' 
              # After step, check if LoRA parameters changed
              for name, param in self.pipe.transformer.named_parameters():
                  if "lora" in name and param.requires_grad:
                      print(f"{name} - First few values after step: {param.data.flatten()[:5]}")
              '''

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

    def run_inference(self, prompt, num_inference_steps=4, guidance_scale=0):
        """Run inference on the fine-tuned model with a given prompt."""
        image = self.pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        return image
