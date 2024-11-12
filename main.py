from data_preprocessing.data_preprocessor import DataPreprocessor
#from models.sd3_flash import SD3Flash

def main():
  """Main function to run pipeline"""
  # Analyze, visualize, and split dataset
  data_preprocessor = DataPreprocessor("TheFusion21/PokemonCards")
  data_preprocessor.analyze_data()
  data_preprocessor.visualize_data()
  data_preprocessor.reduce_captions()
  data_preprocessor.visualize_data()
  dataset_split = data_preprocessor.remove_columns_split_dataset(column_names=["name", "hp", "set_name"])
  data_preprocessor.create_textual_inversion_dataset(dataset_split["train"])


  # load and setup model
  # model_processor = SD3Flash()
  # model_processor.generate_save_images(dataset_split["test"], 
  #                                       num_images=len(dataset_split["test"]), 
  #                                       batch_size=5,
  #                                       user_emails = ["mkschulz@usc.edu", "alopezlo@usc.edu", "oreynozo@usc.edu"])
   
if __name__ == "__main__":
  main()
