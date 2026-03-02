from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":

    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    data_transformation = DataTransformation()

    #Step 1: Data Transformation
    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved at:", preprocessor_path)
    
    
    #Step 2: Model Training
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print("Final Test R2 Score:", r2_score)