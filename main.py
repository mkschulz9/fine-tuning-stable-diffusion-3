from data_preprocessing.data_preprocessor import DataPreprocessor
from models.sd3_flash import SD3Flash
from models.sd3_flash_lora import SD3FlashLora

def main():
    """Main function to run pipeline"""
    # Analyze, visualize, and split dataset
    data_preprocessor = DataPreprocessor("TheFusion21/PokemonCards")
    data_preprocessor.analyze_data()
    #data_preprocessor.visualize_data()
    data_preprocessor.reduce_captions()
    #data_preprocessor.visualize_data()
    dataset_split = data_preprocessor.remove_columns_split_dataset(column_names=["name", "hp", "set_name"])

    # Initialize SD3FlashLora and load the base model
    lora_model = SD3FlashLora()
    lora_model.load_base_model()
    
    # Apply LoRA and fine-tune the model
    lora_model.apply_lora()
    lora_model.fine_tune(dataset_split["train"], epochs=5, lr=5e-4)
    
    # Save the fine-tuned model
    #lora_model.save_finetuned_model("lora_models/lora_v1")
    
    # Run inference with a sample prompt
    #prompt = "A Pokémon card of Pikachu with lightning effects, in a high-quality trading card style."
    #image = lora_model.run_inference(prompt)
    #image.show()

if __name__ == "__main__":
    main()
