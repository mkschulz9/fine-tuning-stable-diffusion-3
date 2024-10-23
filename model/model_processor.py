import os, torch
from tqdm import tqdm
from datetime import datetime
from diffusers import DiffusionPipeline
from datasets import Dataset
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload


class ModelProcessor:
    """Base class tailored for loading and processing data through HuggingFace diffusion models"""
    def __init__(self, model_id: str):
        """Load model from HuggingFace"""
        torch.set_float32_matmul_precision("high")

        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.float16,  # Mixed precision (faster inference)
            #force_download=True
        ).to("cuda")  # Move model to GPU (faster processing)
        
        self.pipe.set_progress_bar_config(disable=True)

        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)

        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune", fullgraph=True)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="max-autotune", fullgraph=True)

        # self.pipe.enable_attention_slicing() # Enable if we need to save some memory in exchange for small speed decrease (https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/image_variation#diffusers.StableDiffusionImageVariationPipeline.enable_attention_slicing)
        # tips if running out of gpu memory: https://huggingface.co/learn/diffusion-course/en/unit3/2#generating-images-from-text
        
        print(f"\nModel loaded: {model_id}")
    
    def print_compatible_schedulers(self):
        print(self.pipe.scheduler.compatibles)
    
    def gdrive_service(self):
        """Google Drive authentication and service setup"""
        SCOPES = ["https://www.googleapis.com/auth/drive"]
        SERVICE_ACCOUNT_FILE = "./model/gdrive-service.json"

        credentials = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        
        service = build("drive", "v3", credentials=credentials)
        return service
    
    def create_and_share_folder(self, service, folder_name, user_emails, parent_id=None):
        """Create new folder in Google Drive (Cloud Service) and share it with specified users"""
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id] if parent_id else []
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()

        if user_emails:
            for user_email in user_emails:
                user_permission = {
                    'type': 'user',
                    'role': 'writer',
                    'emailAddress': user_email
                }
                service.permissions().create(
                    fileId=folder.get('id'),
                    body=user_permission,
                    fields='id',
                ).execute()

        return folder.get('id')

    def generate_single_image(self, caption, seed=None, user_emails: list[str] = None):
        """Generate images using model's pipeline on prompts in test dataset and saves images to google drive"""
            
        print("\n***GENERATING & SAVING IMAGES***")
        
        now = datetime.now()
        str_current_datetime = now.strftime('%Y%m%d_%H%M%S')
        drive_folder_name = f"generated_imgs_{str_current_datetime}"

        service = self.gdrive_service()
        folder_id = self.create_and_share_folder(service, drive_folder_name, user_emails)

        if seed != None:
            generator = torch.Generator("cuda").manual_seed(seed)
            with torch.no_grad():
                with torch.autocast("cuda"):
                    image = self.pipe(caption, num_inference_steps=20, generator=generator).images[0]
        else:
            with torch.no_grad():
                with torch.autocast("cuda"):
                    image = self.pipe(caption, num_inference_steps=20).images[0]
            
        buf = BytesIO()

        image.save(buf, format='PNG')
        buf.seek(0)
        file_metadata = {
            'name': f'generated_img{str_current_datetime}.png',
            'parents': [folder_id]
        }
        media = MediaIoBaseUpload(buf, mimetype='image/png', resumable=True)
        service.files().create(body=file_metadata,
                                                media_body=media,
                                                fields='id').execute()

        
    def generate_save_images(self, test_dataset: Dataset, num_images: int = None, user_emails: list[str] = None):
      """Generate images using model's pipeline on prompts in test dataset and saves images to google drive"""
          
      print("\n***GENERATING & SAVING IMAGES***")

      now = datetime.now()
      str_current_datetime = now.strftime('%Y%m%d_%H%M%S')
      drive_folder_name = f"generated_imgs_{str_current_datetime}"
      
      #self.pipe.eval()
      service = self.gdrive_service()
      folder_id = self.create_and_share_folder(service, drive_folder_name, user_emails)
            
      if num_images is None: 
        num_images = len(test_dataset)
      
      for idx, sample in enumerate(tqdm(test_dataset, total=num_images)):
        if idx >= num_images:
            break
        caption = sample['caption']
            
        with torch.no_grad():
            with torch.autocast("cuda"):
                image = self.pipe(caption).images[0]
            
        buf = BytesIO()

        image.save(buf, format='PNG')
        buf.seek(0)
        file_metadata = {
            'name': f'generated_img{idx}.png',
            'parents': [folder_id]
        }
        media = MediaIoBaseUpload(buf, mimetype='image/png', resumable=True)
        service.files().create(body=file_metadata,
                                              media_body=media,
                                              fields='id').execute()
