Sustainable AI – Energy Estimator & Prompt Optimizer
📌 Project Overview
The Sustainable AI – Energy Estimator & Prompt Optimizer is an innovative system designed to quantify and minimize the energy consumption associated with executing prompts on large language models (LLMs). By combining Natural Language Processing (NLP), machine learning–based energy prediction, anomaly detection, and prompt optimization, this project empowers researchers, developers, and organizations to adopt sustainable AI practices without compromising model performance or semantic quality.

The motivation stems from the growing environmental impact of large-scale AI systems. As LLMs become more prevalent in research, industry, and consumer applications, their computational requirements — and corresponding carbon footprint — have risen sharply. This project addresses that challenge by:

Measuring: Accurately estimating prompt execution energy in kilowatt-hours (kWh).

Optimizing: Recommending alternative prompts that achieve the same intent with lower energy consumption.

Monitoring: Identifying unusual energy usage patterns for operational awareness and optimization opportunities.

🎯 Key Objectives
Provide an accurate and explainable estimate of prompt-level energy usage.

Offer actionable recommendations for low-energy prompt alternatives.

Detect outlier consumption patterns to highlight potential inefficiencies or misuse.

Deliver results through an intuitive, user-friendly interface that encourages adoption across technical and non-technical users.

🚀 Core Features
1. Prompt-Based Energy Estimation
Accepts user input including:

Prompt text

Number of LLM layers used

Known training time

Expected FLOPs per hour

Computes an energy consumption estimate in kWh using a trained predictive model.

2. Low-Energy Prompt Recommendations
Uses embedding-based semantic similarity to find alternative phrasings.

Prioritizes prompts with lower estimated energy usage while preserving meaning.

3. Anomaly Detection
Employs Isolation Forest or One-Class SVM to identify consumption patterns that are statistically abnormal.

Provides early warnings for unexpected performance or resource spikes.

4. Interactive Graphical Interface
Developed with Streamlit for ease of use.

Allows quick iterations between original and optimized prompts.

Displays detailed prediction results with energy breakdowns.

5. Data Logging & Analytics
Maintains a SQLite-backed database of prompts, predictions, and optimizations.

Enables historical analysis for model improvement and usage pattern discovery.

Component Summary
1. User Interface

Functionality: Input collection and result visualization.

Technologies / Libraries: Streamlit.

2. NLP Module

Functionality: Token parsing, complexity scoring, semantic analysis.

Technologies / Libraries: sentence-transformers, T5, GPT-2, OpenAI embeddings.

3. Energy Prediction Engine

Functionality: Supervised machine learning for kWh estimation.

Technologies / Libraries: scikit-learn (Random Forest, Linear Regression).

4. Anomaly Detection

Functionality: Identify abnormal energy consumption patterns.

Technologies / Libraries: Isolation Forest, One-Class SVM.

5. Prompt Optimization

Functionality: Suggest low-energy alternatives while preserving meaning.

Technologies / Libraries: Fine-tuned LLMs, semantic similarity search.

6. Data Logging

Functionality: Store user inputs, predictions, and optimizations for tracking and analysis.

Technologies / Libraries: SQLite.

📊 Implementation Progress
Frontend GUI: Completed; Streamlit-based interface functional.

NLP Module: Token counting and complexity scoring implemented; sentence-transformer integration ongoing.

Energy Prediction: Baseline Linear Regression and Random Forest models trained and validated on sample datasets.

Anomaly Detection: Isolation Forest prototype implemented with functional flagging.

Prompt Optimization: Semantic similarity search implemented; paraphrasing model fine-tuning planned.

Data Logging: SQLite backend integrated for persistent storage.

📈 Roadmap / Next Steps
Fine-tune prompt simplification with T5 or GPT-2 for better paraphrasing accuracy.

Improve energy prediction accuracy with larger and more diverse datasets.

Fully integrate anomaly detection into the user interface.

Enhance recommendation ranking to balance energy efficiency with semantic fidelity.

Implement comprehensive unit and integration testing.

🖥 Installation & Usage
Prerequisites
Python 3.9+

pip / virtualenv recommended

Git

Setup Steps

# Clone repository
git clone https://github.com/SHIRU235/Sustainable_AI.git
cd SustainableAI_Project

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run src/gui/app.py
📚 Technology Stack
Frontend:

Streamlit

NLP Processing:

sentence-transformers

OpenAI embeddings

T5, GPT-2

Machine Learning:

scikit-learn (Random Forest, Linear Regression, Isolation Forest)

Data Storage & Utilities:

SQLite


pandas, numpy, matplotlib
