import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import PeftModel
import huggingface_hub
from huggingface_hub import HfApi

class TextualInversion:
    def __init__(self, model_name, peft_model_name, scheduler_name, device='cuda'):
        self.device = device

        # Load Tokenizer and Text Encoder
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        ).to(self.device)

        # Load UNet Model
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            torch_dtype=torch.float16,
        ).to(self.device)

        # Apply LoRA using PEFT
        self.unet = PeftModel.from_pretrained(self.unet, peft_model_name).to(self.device)

        # Load VAE (Decoder part is used)
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=torch.float16,
        ).to(self.device)

        # Load Scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            scheduler_name,
            subfolder="scheduler",
        )

    def add_new_token(self, new_token):
        # Add New Token
        num_added_tokens = self.tokenizer.add_tokens(new_token)
        if num_added_tokens == 0:
            print(f"The token {new_token} already exists in the tokenizer.")
        else:
            # Resize Token Embeddings
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            # Initialize New Token's Embedding
            token_id = self.tokenizer.convert_tokens_to_ids(new_token)
            with torch.no_grad():
                self.text_encoder.get_input_embeddings().weight[token_id] = torch.randn(
                    self.text_encoder.config.hidden_size, device=self.device
                )
            self.token_id = token_id
            self.new_token = new_token

    def prepare_training(self):
        # Freeze Model Parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Only the embedding of the new token will be trained
        self.embedding_layer = self.text_encoder.get_input_embeddings()
        self.embedding_layer.weight[self.token_id].requires_grad = True

    def train(self, image_paths, prompts, num_epochs=100, lr=1e-4, batch_size=1):
        # Create Dataset and DataLoader
        dataset = TextualInversionDataset(image_paths, prompts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare Optimizer and Loss Function
        optimizer = torch.optim.AdamW([self.embedding_layer.weight[self.token_id]], lr=lr)
        loss_function = torch.nn.MSELoss()

        # Set models to eval mode (since we're not training them)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Training Loop
        for epoch in range(num_epochs):
            for step, (images, inputs) in enumerate(dataloader):
                optimizer.zero_grad()

                # Move data to device
                images = images.to(self.device).half()
                inputs = {key: value.squeeze(1).to(self.device) for key, value in inputs.items()}

                # Encode the images to latent space
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample() * 0.18215  # Scaling factor

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(**inputs).last_hidden_state

                # Sample noise
                noise = torch.randn_like(latents)

                # Get the timesteps
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=self.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute loss
                loss = loss_function(noise_pred, noise)

                # Backpropagation
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    def save_model(self, repo_id):
        # Save only the modified tokenizer and text encoder
        self.tokenizer.save_pretrained(repo_id)
        self.text_encoder.save_pretrained(repo_id)

        # Push to Hugging Face Hub
        api = HfApi()
        api.upload_folder(
            folder_path=repo_id,
            repo_id=repo_id,
            repo_type="model",
        )

class TextualInversionDataset(Dataset):
    def __init__(self, image_paths, prompts, tokenizer):
        self.image_paths = image_paths
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        # Get prompt and tokenize
        prompt = self.prompts[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=self.tokenizer.model_max_length)
        return image, inputs
