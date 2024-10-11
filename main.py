from data_preprocessing.data_preprocessor import DataPreprocessor
from model.model_processor import ModelProcessor

def main():
  """Main function to run pipeline"""
  # Analyze, visualize, and split dataset
  data_preprocessor = DataPreprocessor("TheFusion21/PokemonCards")
  data_preprocessor.analyze_data()
  data_preprocessor.visualize_data()
  dataset_split = data_preprocessor.remove_columns_split_dataset(column_names=["name", "hp", "set_name"])

  # load and setup model
  #model_processor = ModelProcessor("stabilityai/stable-diffusion-3-medium-diffusers")
  #model_processor.generate_images(dataset_split["test"], num_images=5)
  # setup metrics and evaluation loop
  # evaluate and record results
  
if __name__ == "__main__":
  main()