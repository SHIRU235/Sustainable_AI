# import os

# folders = [
#     "data/raw", "data/processed", "data/synthetic",
#     "model/energy_predictor", "model/nlp_transformer", "model/anomaly_detector", "model/prompt_optimizer",
#     "src/gui", "src/nlp", "src/prediction", "src/optimization", "src/anomaly", "src/utils",
#     "reports/visualizations",
#     "documentation/readme_images",
#     "notebooks/training_notebooks", "notebooks/experiments",
#     "tests"
# ]

# for folder in folders:
#     os.makedirs(folder, exist_ok=True)

# # Create base files
# files = [
#     "run.sh",
#     "src/gui/app.py", "src/gui/layout.py",
#     "src/nlp/parser.py", "src/nlp/complexity_score.py", "src/nlp/simplifier.py",
#     "src/prediction/estimator.py", "src/optimization/recommender.py", "src/anomaly/detector.py",
#     "src/utils/logger.py", "src/utils/config.py",
#     "tests/test_nlp.py", "tests/test_predictor.py", "tests/test_gui.py"
# ]

# for file in files:
#     with open(file, "w"): pass  # create empty files


import pandas as pd

# Define mock data
data = pd.DataFrame({
    'num_layers': [2, 4, 6, 8, 10],
    'training_hours': [1, 2, 3, 4, 5],
    'flops_per_hour': [10, 20, 40, 60, 90],
    'energy_consumption': [0.5, 1.3, 2.8, 4.5, 6.9]  # Renamed to match your model's expected column
})

# Save to CSV
csv_path = r"E:\SustainableAI_FinalProject\data\synthetic\energy_data.csv"
data.to_csv(csv_path, index=False)

print(f"✅ Synthetic data saved to: {csv_path}")