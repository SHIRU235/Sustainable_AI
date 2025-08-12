# import sys
# import os
# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from joblib import load

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# from src.optimization.recommender import PromptOptimizer
# from src.prediction.estimator import predict_energy
# from src.anomaly.detector import AnomalyDetector
# from src.nlp.complexity_score import extract_features



   
# def main():

#     # --- Page Config ---
#     st.set_page_config(page_title="GreenMind", layout="wide")

#     # --- Custom CSS for Styling ---
#     st.markdown("""
#         <style>
#             .stApp {
#                 background-image: url('https://res.cloudinary.com/dykxtkzm8/image/upload/v1754630959/WhatsApp_Image_2025-08-08_at_10.49.50_AM_gv5cm2.jpg');
#                 background-size: cover;
#                 background-attachment: fixed;
#             }
#             textarea {
#                 background-color: #f59e0b !important;
#                 color: black !important;
#                 font-weight: 500 !important;
#             }
#             .stButton > button {
#                 font-size: 16px;
#                 font-weight: bold;
#                 border-radius: 8px;
#                 padding: 10px 24px;
#                 margin: 10px 10px 10px 0;
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     # --- Title and Description ---
#     st.markdown("## GreenMind: Energy-Efficient Prompt and Context Engineering")
#     st.caption("Check the predicted Energy Consumption and hit **Improve** to see a more energy-efficient Prompt.")

#     # --- Layout ---
#     col_left, col_right = st.columns([2, 1])

#     with col_right:
#         st.markdown("### Enter prompt here:")
#         prompt = st.text_area(" ", placeholder="""Role -----------------------
#     I am... Lorem ipsum dolor sit amet
#     You are... Consectetur adipiscing elit

#     Context --------------
#     Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

#     Expectations-------
#     Ut enim ad minim veniam""", height=220, label_visibility='collapsed')
        
#         from src.nlp.complexity_score import compute_token_count, compute_readability



#         # Parameter Inputs
#         st.markdown("### Set parameters:")
#         num_layers = st.number_input("# Layers", min_value=1, value=4)
#         training_hours = st.number_input("Training time (hrs)", min_value=1, value=2)
#         flops_per_hour = st.number_input("FLOPs/hr.", min_value=1e5, value=1e20, step=1e18, format="%.2e")

#     st.markdown("""
#         <style>
#         div.stButton > button:first-child {
#             background-color:  #001f3f; /* Green */
#             color: white;
#             border-radius: 10px;
#             height: 50px;
#             font-size: 18px;
#         }
#         div.stButton > button:first-child:hover {
#             background-color: #004080; /*darker green */
#             color: white;
#         }
#         </style>
#     """, unsafe_allow_html=True)
#     # Submit/Improve Buttons
#     col_submit, col_improve = st.columns([1, 1])
#     submit_clicked = col_submit.button("Submit")
#     improve_clicked = col_improve.button("Improve", use_container_width=True)

#     # --- Prediction Logic ---
#     if submit_clicked:
#         st.subheader("🔋 Estimated Energy Consumption")

#             # --- NLP Analysis ---
#         tokens = compute_token_count(prompt)
#         readability = compute_readability(prompt)

#         st.subheader(" NLP Analysis")
#         st.markdown(f"- **Token Count:** {tokens}")
#         st.markdown(f"- **readability_score:** {readability:.2f}")

#         input_data = {
#             "num_layers": num_layers,
#             "training_hours": training_hours,
#             "flops_per_hour": flops_per_hour,
#             "token_count": tokens,
#             "readability_score": readability
#         }

#         model_path = os.path.join("model", "energy_predictor.pkl")

#         try:
#             prediction = predict_energy(model_path, input_data)
#             st.success(f"⚡ Estimated Energy Consumption: **{prediction:.2f} kWh**")
#         except Exception as e:
#             st.error(f"❌ Prediction failed: {e}")

#         # Optional Mock Graph
#         st.markdown("### 📈 Energy Prediction vs Actual (Mocked)")
#         actual = np.array([1, 2, 3, 4, 5, 6, 7])
#         predicted = actual + np.random.normal(0, 0.3, size=actual.shape)
  
 

#         try:
#             # Load training dataset
#             training_data_path = r"E:\SustainableAiProject\data\processed\processed_data.csv"
#             df = pd.read_csv(training_data_path)

#             # Load trained model
#             model_path = os.path.join("model", "energy_predictor.pkl")
#             model = load(model_path)

#             # Separate features and target
#             X = df.drop(columns=['energy_consumption'])
#             y_actual = df['energy_consumption']

#             # Predict using the trained model
#             y_pred = model.predict(X)

#             # Example: pick the 3rd data point as the prediction to highlight
#             pred_index = 2
#             pred_layer = df['num_layers'].iloc[pred_index]
#             pred_value = y_pred[pred_index]

#             # Plot
#             fig, ax = plt.subplots()
#             ax.plot(df['num_layers'], y_actual, 'o-', label="Actual")
#             ax.plot(df['num_layers'], y_pred, 's--', label="Predicted")  # square markers

#             # Highlight one prediction point
#             # ax.scatter(pred_layer, pred_value, color='cyan', edgecolors='black', s=150, marker='s', zorder=5)

#             # Highlight one prediction point
#             ax.scatter(pred_layer, pred_value, color='cyan', edgecolors='black', s=150, marker='s', zorder=5)

#             # Draw vertical line from x-axis to prediction
#             ax.plot([pred_layer, pred_layer], [0, pred_value], color='orange', linewidth=2)

#             # Draw horizontal line from y-axis to prediction
#             ax.plot([0, pred_layer], [pred_value, pred_value], color='orange', linewidth=2)
#             # Add annotation with background box
#             ax.annotate(
#                 "Prediction",
#                 xy=(pred_layer, pred_value),
#                 xytext=(pred_layer + 1, pred_value + 0.5),
#                 arrowprops=dict(facecolor='cyan', shrink=0.05, width=1, headwidth=6),
#                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
#             )
#     #         plt.plot([pred_x, pred_x], [0, pred_y], color='orange', linewidth=2)

#     # # Draw horizontal line from y-axis to prediction
#     #         plt.plot([0, pred_x], [pred_y, pred_y], color='orange', linewidth=2)
#             ax.set_xlabel("Number of Layers")
#             ax.set_ylabel("Energy Consumption (kWh)")
#             ax.legend()
#             ax.grid(True)

#             st.pyplot(fig)

#         except Exception as e:
#             st.error(f"❌ Could not plot Actual vs Predicted: {e}")
#                 # --- Anomaly Detection ---
#         features = extract_features(prompt)
#         flops_per_layer = flops_per_hour / num_layers
#         sample = [prediction, features["token_count"], flops_per_layer]
#             # Load and run anomaly detector
#         anomaly_detector = AnomalyDetector()
#         anomaly_detector.fit([
#                 [0.3, 45, 1e17],
#                 [0.5, 55, 2e17],
#                 [0.7, 60, 3e17],
#                 [0.4, 50, 1.5e17],
#                 [0.6, 58, 2.5e17],
#                 ])
#         is_anomaly, reason = anomaly_detector.detect(sample)

#         if is_anomaly:
                    
#                     st.error("🕵️ Anomaly Detected: This prompt may consume excessive energy.")
#                     st.info(f"Reason: {reason}")

#     # --- Prompt Optimization ---
#     if improve_clicked:
#         if prompt.strip() == "":
#             st.warning("Please enter a prompt before clicking Improve.")
#         else:
#             st.info("Optimizing your prompt... please wait ⏳")
#             optimizer = PromptOptimizer()
#             suggestions = optimizer.suggest_prompts(prompt, num_variants=10, top_k=5)

#             st.success("🔧 Here are improved, energy-efficient prompts:")
#             for idx, (variant, score) in enumerate(suggestions, 1):
#                 st.markdown(f"**{idx}.** `{variant}`  &nbsp; _(Similarity: {score.item():.4f})_")



# if __name__ == "__main__":
#     main()
import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import logging

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,  # Captures DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning")
logging.error("This is an error")
logging.critical("This is critical")


# Import UI functions
from layout import apply_custom_css, show_header, get_prompt_input, get_parameter_inputs, action_buttons

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.optimization.recommender import PromptOptimizer
from src.prediction.estimator import predict_energy
from src.anomaly.detector import AnomalyDetector
from src.nlp.complexity_score import extract_features, compute_token_count, compute_readability


def main():
    logging.info("Streamlit app started")
    st.set_page_config(page_title="GreenMind", layout="wide")

    # Apply styling
    apply_custom_css()
    logging.info("Custom CSS applied")

    # Header
    show_header()

    # Layout
    col_left, col_right = st.columns([2, 1])
    with col_right:
        prompt = get_prompt_input()
        num_layers, training_hours, flops_per_hour = get_parameter_inputs()

    logging.info(f"User input - num_layers: {num_layers}, training_hours: {training_hours}, flops_per_hour: {flops_per_hour}")

    # Buttons
    submit_clicked, improve_clicked = action_buttons()

    # --- Prediction Logic ---
    if submit_clicked:
        logging.info("Submit button clicked")
        st.subheader("🔋 Estimated Energy Consumption")

        try:
            tokens = compute_token_count(prompt)
            readability = compute_readability(prompt)
            logging.info(f"Token count: {tokens}, Readability: {readability}")
        except Exception as e:
            logging.error(f"Failed NLP analysis: {e}")
            st.error(f"❌ NLP analysis failed: {e}")
            return

        st.subheader(" NLP Analysis")
        st.markdown(f"- **Token Count:** {tokens}")
        st.markdown(f"- **Readability Score:** {readability:.2f}")

        input_data = {
            "num_layers": num_layers,
            "training_hours": training_hours,
            "flops_per_hour": flops_per_hour,
            "token_count": tokens,
            "readability_score": readability
        }
        logging.info(f"Prediction input data: {input_data}")

        model_path = os.path.join("model", "E:\SustainableAiProject\model\energy_predictor\energy_predictor.pkl")
        try:
            prediction = predict_energy(model_path, input_data)
            st.success(f"⚡ Estimated Energy Consumption: **{prediction:.2f} kWh**")
            logging.info(f"Prediction successful: {prediction}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            logging.error(f"Prediction failed: {e}")

        # --- Plot Actual vs Predicted ---
        try:
            df = pd.read_csv(r"E:\SustainableAiProject\data\processed\processed_data.csv")
            model = load(model_path)

            X = df.drop(columns=['energy_consumption'])
            y_actual = df['energy_consumption']
            y_pred = model.predict(X)

            pred_index = 2
            pred_layer = df['num_layers'].iloc[pred_index]
            pred_value = y_pred[pred_index]

            fig, ax = plt.subplots()
            ax.plot(df['num_layers'], y_actual, 'o-', label="Actual")
            ax.plot(df['num_layers'], y_pred, 's--', label="Predicted")
            ax.scatter(pred_layer, pred_value, color='cyan', edgecolors='black', s=150, marker='s', zorder=5)
            ax.plot([pred_layer, pred_layer], [0, pred_value], color='orange', linewidth=2)
            ax.plot([0, pred_layer], [pred_value, pred_value], color='orange', linewidth=2)
            ax.annotate("Prediction",
                        xy=(pred_layer, pred_value),
                        xytext=(pred_layer + 1, pred_value + 0.5),
                        arrowprops=dict(facecolor='cyan', shrink=0.05, width=1, headwidth=6),
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
            ax.set_xlabel("Number of Layers")
            ax.set_ylabel("Energy Consumption (kWh)")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
            logging.info("Plot generated successfully")
        except Exception as e:
            st.error(f"❌ Could not plot Actual vs Predicted: {e}")
            logging.error(f"Plot generation failed: {e}")

        # --- Anomaly Detection ---
        try:
            features = extract_features(prompt)
            flops_per_layer = flops_per_hour / num_layers
            sample = [prediction, features["token_count"], flops_per_layer]

            anomaly_detector = AnomalyDetector()
            anomaly_detector.fit([
                [0.3, 45, 1e17],
                [0.5, 55, 2e17],
                [0.7, 60, 3e17],
                [0.4, 50, 1.5e17],
                [0.6, 58, 2.5e17],
            ])
            is_anomaly, reason = anomaly_detector.detect(sample)

            if is_anomaly:
                st.error("🕵️ Anomaly Detected: This prompt may consume excessive energy.")
                st.info(f"Reason: {reason}")
                logging.warning(f"Anomaly detected - Reason: {reason}")
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")

    # --- Prompt Optimization ---
    if improve_clicked:
        logging.info("Improve button clicked")
        if prompt.strip() == "":
            st.warning("Please enter a prompt before clicking Improve.")
            logging.warning("Improve clicked with empty prompt")
        else:
            st.info("Optimizing your prompt... please wait ⏳")
            try:
                optimizer = PromptOptimizer()
                suggestions = optimizer.suggest_prompts(prompt, num_variants=10, top_k=5)
                st.success("🔧 Here are improved, energy-efficient prompts:")
                for idx, (variant, score) in enumerate(suggestions, 1):
                    st.markdown(f"**{idx}.** `{variant}`  _(Similarity: {score.item():.4f})_")
                logging.info(f"Prompt optimization successful: {len(suggestions)} suggestions generated")
            except Exception as e:
                st.error(f"❌ Prompt optimization failed: {e}")
                logging.error(f"Prompt optimization failed: {e}")


if __name__ == "__main__":
    main()