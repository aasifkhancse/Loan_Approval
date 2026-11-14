Loan Approval Prediction

A Machine Learning classification project that analyzes applicant information and predicts whether a loan application will be approved or rejected.

Project Overview

The Loan Approval Prediction project uses machine learning algorithms to predict loan approval outcomes based on applicant financial and personal information.

The project follows an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and model deployment preparation.

Multiple classification algorithms are compared using different evaluation metrics to identify a suitable predictive model.

Objectives

- Analyze loan applicant data.
- Handle missing values and inconsistent data.
- Perform exploratory data analysis (EDA).
- Encode categorical variables.
- Apply feature scaling.
- Perform feature engineering.
- Train and compare multiple classification models.
- Evaluate model performance.
- Save the trained model and preprocessing objects.

Project Workflow

1. Data Collection and Inspection
   
   - Load the loan approval dataset using Pandas.
   - Inspect data types, statistical summaries, and missing values.

2. Data Preprocessing
   
   - Handle missing numerical values using mean imputation.
   - Handle missing categorical values using mode imputation.
   - Remove unnecessary applicant ID columns.
   - Encode categorical variables.

3. Exploratory Data Analysis
   
   - Analyze loan approval distributions.
   - Explore applicant income and financial characteristics.
   - Visualize relationships between features and loan approval.

4. Feature Engineering
   
   - Create squared features for "DTI_Ratio" and "Credit_Score".
   - Apply one-hot encoding.
   - Analyze feature correlations.

5. Model Training
   
   - Split the dataset into training and testing sets (80:20).
   - Apply StandardScaler where required.
   - Train multiple classification algorithms.

6. Model Evaluation
   
   - Compare models using accuracy, precision, recall, and F1-score.
   - Generate confusion matrices.
   - Identify a suitable model based on evaluation results.

7. Model Saving
   
   - Export the trained Random Forest model.
   - Save the scaler, one-hot encoder, and feature-column list using Pickle.

Machine Learning Algorithms

The project explores the following classification algorithms:

- Logistic Regression
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- Bagging Classifier
- AdaBoost Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Multi-Layer Perceptron (MLP)
- XGBoost
- LightGBM

Evaluation Metrics

The models are evaluated using:

Metric| Description
Accuracy| Overall percentage of correct predictions
Precision| Percentage of predicted approvals that are actually approved
Recall| Percentage of actual approvals correctly identified
F1-Score| Harmonic mean of precision and recall
Confusion Matrix| Shows correct and incorrect predictions for each class

Model Selection: Random Forest was identified as the preferred model based on precision in the notebook's initial model comparison. Additional model results, including LightGBM, are also explored.

Final performance should be verified by running the complete notebook.

Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Pickle

Repository Structure

Loan-Approval-Prediction/
│
├── LoanApproverNoteBook.ipynb
├── README.md
├── LICENSE
│
├── loan_approval_model.pkl
├── scaler.pkl
├── onehot_encoder.pkl
└── feature_columns.pkl

Note: The ".pkl" files are generated when the final notebook cell is executed. They may not be available in a fresh repository clone.

Dataset

The project uses a dataset named:

"loan_approval_data.csv"

The notebook currently reads the dataset from Google Drive:

df = pd.read_csv(
    "/content/drive/MyDrive/My Work/loan_approval_data.csv"
)

Update the file path according to your local environment.

Note: The dataset is not included in this repository.

Installation

Clone the repository:

git clone https://github.com/aasifkhancse/Loan_Approval.git

Navigate to the project directory:

cd Loan_Approval

Install the required dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm

How to Run

1. Clone or download the repository.
2. Install the required Python libraries.
3. Download or provide the loan approval dataset.
4. Update the dataset path in the notebook.
5. Open "LoanApproverNoteBook.ipynb" using Jupyter Notebook or Google Colab.
6. Execute all cells sequentially.
7. Review the model evaluation results.
8. Execute the final cell to generate and save the model artifacts.

Model Artifacts

The notebook saves the following files:

File| Purpose
"loan_approval_model.pkl"| Trained Random Forest model
"scaler.pkl"| Fitted StandardScaler
"onehot_encoder.pkl"| Fitted OneHotEncoder
"feature_columns.pkl"| Feature names and training order

These artifacts can be used to prepare applicant data and generate predictions, provided inference follows the same preprocessing and feature order used during training.

Limitations and Future Improvements

- Use Scikit-learn Pipelines and ColumnTransformer to prevent data leakage.
- Perform hyperparameter tuning to improve model performance.
- Apply cross-validation for more reliable evaluation.
- Perform feature importance analysis.
- Evaluate fairness and potential bias in loan approval predictions.
- Develop a web application using Flask or Streamlit.
- Deploy the trained model using a suitable cloud platform.

Disclaimer

This project is developed for educational and machine learning practice purposes.

It is not intended to be used as a production loan-underwriting system. Real-world lending decisions require proper validation, fairness assessment, explainability, security controls, and human oversight.

Only load Pickle files from trusted sources.

License

This project is licensed under the MIT License.

Author

Aasif Khan 
