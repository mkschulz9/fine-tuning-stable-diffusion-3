import os
from glob import glob
from itertools import product
from data_preprocessing.data_preprocessor import DataPreprocessor
from models.sd3_flash import SD3Flash
from textual_inversion import TextualInversion

def main():
  """Main function to run pipeline"""
  # Analyze, visualize, and split dataset
  data_preprocessor = DataPreprocessor("TheFusion21/PokemonCards")
  data_preprocessor.analyze_data()
  data_preprocessor.visualize_data()
  data_preprocessor.reduce_captions()
  data_preprocessor.visualize_data()
  dataset_split = data_preprocessor.remove_columns_split_dataset(column_names=["name", "hp", "set_name"])

  # Initialization
  model_name = "stabilityai/stable-diffusion-3-medium-diffusers"
  peft_model_name = "jasperai/flash-sd3"
  scheduler_name = model_name
  device = 'cuda'

  textual_inversion = TextualInversion(model_name, peft_model_name, scheduler_name, device)

  # Add New Token
  new_token = "S*"
  textual_inversion.add_new_token(new_token)
  textual_inversion.prepare_training()

  # Path to your image folder
  image_folder = 'path/to/your/images'

  # Get list of image paths
  image_paths = glob(os.path.join(image_folder, '*'))
  image_paths.sort()

  # Create prompts
  imagenet_templates_small = [
      "a photo of a {}",
      "a rendering of a {}",
      "a cropped photo of the {}",
      "the photo of a {}",
      "a photo of a clean {}",
      "a photo of a dirty {}",
      "a dark photo of the {}",
      "a photo of my {}",
      "a photo of the cool {}",
      "a close-up photo of a {}",
      "a bright photo of the {}",
      "a cropped photo of a {}",
      "a photo of the {}",
      "a good photo of the {}",
      "a photo of one {}",
      "a close-up photo of the {}",
      "a rendition of the {}",
      "a photo of the clean {}",
      "a rendition of a {}",
      "a photo of a nice {}",
      "a good photo of a {}",
      "a photo of the nice {}",
      "a photo of the small {}",
      "a photo of the weird {}",
      "a photo of the large {}",
      "a photo of a cool {}",
      "a photo of a small {}",
  ]

  imagenet_style_templates_small = [
      "a painting in the style of {}",
      "a rendering in the style of {}",
      "a cropped painting in the style of {}",
      "the painting in the style of {}",
      "a clean painting in the style of {}",
      "a dirty painting in the style of {}",
      "a dark painting in the style of {}",
      "a picture in the style of {}",
      "a cool painting in the style of {}",
      "a close-up painting in the style of {}",
      "a bright painting in the style of {}",
      "a cropped painting in the style of {}",
      "a good painting in the style of {}",
      "a close-up painting in the style of {}",
      "a rendition in the style of {}",
      "a nice painting in the style of {}",
      "a small painting in the style of {}",
      "a weird painting in the style of {}",
      "a large painting in the style of {}",
  ]

  all_templates = imagenet_templates_small + imagenet_style_templates_small
  prompts = [template.format(new_token) for template in all_templates]

  # Get image paths
  image_folder = '../datasets/5_imgs'
  image_paths = glob(os.path.join(image_folder, '*'))
  image_paths.sort()

  # Create image-prompt pairs
  # Option 1: All combinations
  image_prompt_pairs = list(product(image_paths, prompts))

  # Option 2: Random selection (e.g., 5 prompts per image)
  # random.seed(42)  # For reproducibility
  # prompts_per_image = 5
  # image_prompt_pairs = []
  # for image_path in image_paths:
  #     selected_prompts = random.sample(prompts, k=prompts_per_image)
  #     for prompt in selected_prompts:
  #         image_prompt_pairs.append((image_path, prompt))

  # Initialize and prepare the TextualInversion class
  textual_inversion = TextualInversion(model_name, peft_model_name, scheduler_name, device)
  textual_inversion.add_new_token(new_token)
  textual_inversion.prepare_training()

  # Train the model
  textual_inversion.train(image_prompt_pairs, num_epochs=100, lr=1e-4, batch_size=1)

  # Save the Fine-Tuned Model to Hugging Face
  repo_id = "mkschulz9/flash-sd3-textual-inversion"

  textual_inversion.save_model(repo_id)
 
  # load and setup model
  # model_processor = SD3Flash()
  # model_processor.generate_save_images(dataset_split["test"], 
  #                                       num_images=len(dataset_split["test"]), 
  #                                       batch_size=5,
  #                                       user_emails = ["mkschulz@usc.edu", "alopezlo@usc.edu", "oreynozo@usc.edu"])
   
if __name__ == "__main__":
  main()
