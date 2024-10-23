import os, torch, json
from tqdm import tqdm
from datetime import datetime
from datasets import Dataset
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler
from peft import PeftModel


class SD3Flash:
    """Base class tailored for loading and processing data through HuggingFace diffusion models"""
    def __init__(self):
        """Load model from HuggingFace"""
        # Load LoRA
        transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        transformer = PeftModel.from_pretrained(transformer, "jasperai/flash-sd3")


        # Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            transformer=transformer,
            torch_dtype=torch.float16,
            text_encoder_3=None,
            tokenizer_3=None
        )

        # Scheduler
        self.pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
          "stabilityai/stable-diffusion-3-medium-diffusers",
          subfolder="scheduler",
        )

        self.pipe.to("cuda")

        
        print(f"\nModel loaded")
        
    def generate_save_images(self, test_dataset: Dataset, num_images: int = None, user_emails: list[str] = None):
      """Generate images using model's pipeline on prompts in test dataset and saves images to google drive"""
      def gdrive_service():
        """Google Drive authentication and service setup"""
        SCOPES = ["https://www.googleapis.com/auth/drive"]
        SERVICE_ACCOUNT_FILE = "./models/gdrive-service.json"

        credentials = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        
        service = build("drive", "v3", credentials=credentials)
        return service
    
      def create_and_share_folder(service, folder_name, user_emails, parent_id=None):
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
    
      def create_subfolder(service, parent_folder_id, subfolder_name):
        """Create a subfolder within a given parent folder."""
        subfolder_metadata = {
            'name': subfolder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        subfolder = service.files().create(body=subfolder_metadata, fields='id').execute()
        return subfolder.get('id')
    
      print("\n***GENERATING, SAVING IMAGES & IDs***")
      
      now = datetime.now()
      str_current_datetime = now.strftime('%Y%m%d_%H%M%S')
      drive_folder_name = f"generated_imgs_{str_current_datetime}"
      imgs_subfolder_name = 'generated_imgs'
      
      #self.pipe.eval()
      service = gdrive_service()
      drive_folder_id = create_and_share_folder(service, drive_folder_name, user_emails)
      imgs_subfolder_id = create_subfolder(service, drive_folder_id, imgs_subfolder_name)
      
      if num_images is None: 
        num_images = len(test_dataset)
        
      img_ids = {}
      
      for idx, sample in enumerate(tqdm(test_dataset, total=num_images)):
        if idx >= num_images:
          break
        sample_caption = sample['caption']
        sample_id = sample['id']
            
        with torch.no_grad():
            with torch.autocast("cuda"):
                image = self.pipe(sample_caption, num_inference_steps=8, guidance_scale=0).images[0]
            
        buf = BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        
        img_metadata = {
          'name': f'generated_img{idx}.png',
          'parents': [imgs_subfolder_id]
        }
        media = MediaIoBaseUpload(buf, mimetype='image/png', resumable=True)
        service.files().create(body=img_metadata,
                               media_body=media,
                               fields='id').execute()
        img_ids[idx] = sample_id
        
      id_content = json.dumps(img_ids, indent=4)
      id_buf = BytesIO(id_content.encode('utf-8'))
      id_buf.seek(0)
      
      id_file_metadata = {
        'name': 'img_ids.json',
        'parents': [drive_folder_id]
      }

      id_media = MediaIoBaseUpload(id_buf, mimetype='application/json', resumable=True)
      service.files().create(body=id_file_metadata, media_body=id_media, fields='id').execute()