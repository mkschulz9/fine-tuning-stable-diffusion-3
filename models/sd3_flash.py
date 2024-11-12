import os, torch, json, tracemalloc, time, sys
from tqdm import tqdm
from datetime import datetime
from datasets import Dataset
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler, models
from peft import PeftModel

project_root = "/content/fine-tuning-stable-diffusion-3/models"
sys.path.append(project_root)

from utils import gdrive_service, create_and_share_folder, create_subfolder

class SD3Flash:
    """Base class tailored for loading and processing data through HuggingFace diffusion models"""
    def __init__(self):
        """Load model from HuggingFace"""
        
        tracemalloc.start()
        start_current, start_peak = tracemalloc.get_traced_memory()
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
        self.pipe.set_progress_bar_config(disable=True)
        end_current, end_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.memory_used = end_current - start_current
        print(f"\nModel loaded")
        
    def generate_save_images(self, test_dataset: Dataset, num_images: int = None, user_emails: list[str] = None, batch_size: int = 20):
      print("\n***GENERATING, SAVING IMAGES & IDs***")
      
      now = datetime.now()
      str_current_datetime = now.strftime('%Y%m%d_%H%M%S')
      drive_folder_name = f"generated_imgs_{str_current_datetime}"
      imgs_subfolder_name = 'generated_imgs'
      
      service = gdrive_service()
      drive_folder_id = create_and_share_folder(service, drive_folder_name, user_emails)
      imgs_subfolder_id = create_subfolder(service, drive_folder_id, imgs_subfolder_name)
      
      if num_images is None: 
        num_images = len(test_dataset)
        
      data = {}
      data["model_load_memory_used"] = self.memory_used

      # Create data.json initially
      data_content = json.dumps(data, indent=4)
      data_buf = BytesIO(data_content.encode('utf-8'))
      data_buf.seek(0)
      
      data_file_metadata = {
        'name': 'data.json',
        'parents': [drive_folder_id]
      }
      data_media = MediaIoBaseUpload(data_buf, mimetype='application/json', resumable=True)
      file = service.files().create(body=data_file_metadata, media_body=data_media, fields='id').execute()
      data_file_id = file.get('id')

      for idx, sample in enumerate(tqdm(test_dataset, total=num_images, disable=True)):
        if idx >= num_images:
          break
        sample_caption = sample['caption']
        sample_id = sample['id']
        
        start_time = time.time()
        with torch.no_grad():
            with torch.autocast("cuda"):
                image = self.pipe(sample_caption, num_inference_steps=4, guidance_scale=0).images[0]
        end_time = time.time()
        elapsed_time_secs = (end_time - start_time)

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
        data[idx] = {"id": sample_id, "inference_time": elapsed_time_secs}

        if (idx + 1) % batch_size == 0 or (idx + 1) == num_images:
          data_content = json.dumps(data, indent=4)
          data_buf = BytesIO(data_content.encode('utf-8'))
          data_buf.seek(0)
          
          data_media = MediaIoBaseUpload(data_buf, mimetype='application/json', resumable=True)
          service.files().update(fileId=data_file_id, body={'name': 'data.json'}, media_body=data_media).execute()