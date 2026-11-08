# Comparison of Classification Models for Breast Cancer Prediction

## a. Problem statement

The objective of this project is to train and compare five classification algorithms for predicting whether a breast tumour is malignant or benign.

All five models were evaluated on the same test dataset using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

I also developed a Streamlit application that loads the saved models, accepts labelled test data in CSV format, calculates the evaluation metrics, and presents the results without training the models again.

## b. Dataset description

I used the Breast Cancer Wisconsin Diagnostic dataset available through scikit-learn. The original dataset is from the UCI Machine Learning Repository.

The dataset details are:

- **Number of records:** 569
- **Number of input features:** 30 numerical features
- **Target column:** `target`
- **Target encoding:** 0 represents malignant and 1 represents benign
- **Class distribution:** 212 malignant and 357 benign records
- **Training and testing split:** 80% training and 20% testing
- **Test-set size:** 114 records
- **Split method:** Stratified split with `random_state=42`

The dataset satisfies the assignment requirement because it contains more than 500 records and more than 12 input features.

I used a stratified split to maintain approximately the same malignant-to-benign class proportion in both the training and test datasets. I used `random_state=42` to make the dataset split and model results reproducible.

## c. GitHub Repository Link

https://github.com/2026aj/breast-cancer-ml-classification

## d. Models used and comparison

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

All five models were trained using the same training dataset and evaluated using the same 114-record test dataset to ensure a consistent comparison.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9967 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## Performance observations

| ML Model Name | My observation |
|---|---|
| Logistic Regression | Logistic Regression produced an Accuracy of 0.9825 and showed balanced Precision and Recall. It classified 112 of the 114 test records correctly. |
| Decision Tree | Decision Tree was the easiest model to interpret, but it produced the lowest Accuracy, AUC, F1 Score, and MCC among the five models. It classified 103 of the 114 test records correctly. |
| KNN | KNN achieved a Recall of 1.0000 for the positive class, which is benign in this dataset. It correctly identified all 72 benign records, but three malignant records were predicted as benign. |
| Naive Bayes | Naive Bayes was computationally simple and achieved a high AUC of 0.9878. However, its Accuracy and MCC were lower than those of Logistic Regression, KNN, and SVM. |
| Support Vector Machine | SVM achieved the highest AUC of 0.9967. It also tied with Logistic Regression for the highest Accuracy, Precision, Recall, F1 Score, and MCC. |
| **Selected best model** | **Support Vector Machine was the best overall model in this experiment because it achieved the highest AUC while maintaining strong and balanced results across the other evaluation metrics.** |

## Confusion matrix observations

The confusion matrices provide a class-wise view of correct and incorrect predictions.

For the Support Vector Machine, the confusion matrix was:

```text
[[41, 1],
 [ 1, 71]]
```

This means:

- 41 malignant records were correctly predicted as malignant.
- 1 malignant record was incorrectly predicted as benign.
- 1 benign record was incorrectly predicted as malignant.
- 71 benign records were correctly predicted as benign.
- 112 of the 114 test records were classified correctly.

The Support Vector Machine therefore achieved an Accuracy of:

```text
112 / 114 = 0.9825
```

## Repository structure

```text
breast-cancer-ml-classification/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_metrics.csv
|-- .gitignore
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
|   `-- metadata.json
```

## Running the project

### Windows PowerShell

I used the following commands to run and test the project locally on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py --server.address localhost
```

After starting Streamlit, the application can be opened in a browser at:

```text
http://localhost:8501
```

The Streamlit server can be stopped by pressing `Ctrl+C` in the terminal.

### BITS Virtual Lab

The BITS Virtual Lab uses a Linux environment. The following commands can be used to run the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

If the default command in the lab is `python` instead of `python3`, the virtual environment can be created using:

```bash
python -m venv .venv
```

The `train_models.py` script trains and saves the five models. It also generates `test_data.csv` and `model_metrics.csv`.

The `app.py` file loads the saved models and provides the Streamlit evaluation interface.

## Streamlit application features

The Streamlit application provides the following functionality:

- Upload a labelled CSV test file
- Select one of the five saved classification models
- Display a short explanation of the selected algorithm
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Display a confusion matrix for malignant and benign classes
- Display a class-wise classification report
- Preview predictions for the uploaded test records
- Download the prediction results as a CSV file
- Validate required columns
- Detect missing values
- Detect empty uploaded files
- Validate the target labels
- Load saved models without retraining them

## Prediction output

The application allows the prediction results to be downloaded as `predictions.csv`.

The downloaded file contains:

- The original 30 input features
- The actual `target`
- The model-generated `predicted_target`
- The predicted `benign_probability`

The class labels are:

```text
0 = Malignant
1 = Benign
```

The downloaded CSV file can be opened using Microsoft Excel for row-by-row inspection.

## Deployment on Streamlit Community Cloud

The application will be deployed using Streamlit Community Cloud with the following configuration:

- **GitHub repository:** `2026aj/breast-cancer-ml-classification`
- **Branch:** `main`
- **Application entry point:** `app.py`
- **Dependency file:** `requirements.txt`

The deployment procedure is:

1. Push the complete project to the GitHub repository.
2. Sign in to Streamlit Community Cloud using the connected GitHub account.
3. Create a new Streamlit application.
4. Select the `2026aj/breast-cancer-ml-classification` repository.
5. Select the `main` branch.
6. Set the application file path to `app.py`.
7. Deploy the application.
8. Upload `test_data.csv`.
9. Verify the metrics for all five models.

## Live Streamlit Application

Deployment is in progress. The public application URL will be included after the Streamlit Community Cloud deployment is completed.

## Reproducing the results

The `train_models.py` script performs the following operations:

1. Loads the Breast Cancer Wisconsin Diagnostic dataset.
2. Separates the 30 input features and target column.
3. Creates a stratified 80% training and 20% testing split.
4. Trains all five classification models using the same training data.
5. Evaluates every model using the same test data.
6. Calculates Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
7. Saves the trained models in the `model` directory.
8. Saves the feature names and model filenames in `metadata.json`.
9. Generates `test_data.csv`.
10. Generates `model_metrics.csv`.

The Streamlit application does not call the model-training process. It loads the saved `.joblib` files and uses them only for prediction and evaluation.

## Key implementation decisions

- I used `random_state=42` so that the train-test split and evaluation results can be reproduced.
- I used stratification to preserve the malignant-to-benign class proportion in the training and test datasets.
- I used `StandardScaler` with Logistic Regression, KNN, and SVM because these algorithms are affected by differences in feature magnitude.
- I included scaling inside the model pipelines so that the same preprocessing is applied during training and prediction.
- I limited the Decision Tree depth to reduce overfitting.
- I used seven neighbors for the KNN classifier.
- I used an RBF kernel for SVM to support nonlinear class boundaries.
- I saved the complete trained models using joblib.
- I used `st.cache_resource` to avoid repeatedly loading the same model during Streamlit interactions.
- I kept the training process separate from the Streamlit application so that uploaded test data is not used for retraining.
- I added input validation to provide clear messages when the uploaded CSV is incomplete or invalid.

## Conclusion

All five classification models were trained and evaluated successfully using the same test dataset. Logistic Regression and Support Vector Machine produced the highest Accuracy of 0.9825. Support Vector Machine achieved the highest AUC of 0.9967 and was selected as the best overall model in this comparison.

The Streamlit application provides an interactive method for selecting a saved model, uploading test data, viewing evaluation results, inspecting the confusion matrix, and downloading predictions.