import os
import unittest
import tempfile
import pandas as pd
import joblib

from src.prediction.estimator import train_energy_model, predict_energy


class TestEnergyPredictor(unittest.TestCase):

    def setUp(self):
        """Create a temporary CSV for testing"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "dummy_data.csv")
        self.model_path = os.path.join(self.temp_dir.name, "energy_predictor.pkl")

        # Create a small dataset with required columns
        data = {
            'num_layers': [6, 12, 24],
            'training_hours': [10, 20, 30],
            'flops_per_hour': [1e12, 2e12, 3e12],
            'token_count': [500, 1000, 1500],
            'readability_score': [70.0, 60.0, 50.0],
            'energy_consumption': [100.0, 200.0, 300.0]
        }
        pd.DataFrame(data).to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Clean up temp files"""
        self.temp_dir.cleanup()

    def test_train_and_predict(self):
        """Test training and prediction"""
        # Train the model
        model = train_energy_model(self.csv_path, save_model_path=self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(hasattr(model, "predict"))

        # Prepare input features
        features = {
            "num_layers": 8,
            "training_hours": 15,
            "flops_per_hour": 1.5e12,
            "token_count": 800,
            "readability_score": 65.0
        }

        # Predict
        prediction = predict_energy(self.model_path, features)
        self.assertIsInstance(prediction, float)
        # The value should be within a reasonable range based on training
        self.assertGreater(prediction, 0)


if __name__ == "__main__":
    unittest.main()