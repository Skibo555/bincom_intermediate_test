import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# Opening the csv files
test_data = pd.read_csv("loan.csv")
train_data = pd.read_csv("train_data.csv")

# Keep Loan_ID separately so to be able to use it later for Identification purposes
test_data_ids = test_data["Loan_ID"]

# Dropping unwanted column
"""
train_data.drop(["Loan_ID"], axis=1): This command tells pandas to drop the specified columns from the DataFrame. 
Here, ["Loan_ID"] is the column being dropped.
axis=1: This specifies that columns should be dropped (instead of rows, which would be axis=0).
inplace=True: This means the operation is performed in place, modifying the existing DataFrame rather than 
returning a new one with the column removed.

The "Loan_ID" column typically contains unique identifiers for each loan, which are not useful as features for training 
the model.
Including Loan_ID in the model would not provide any predictive power and could potentially lead to over-fitting.
The identifier might be needed later for reporting or submission purposes, but it is not useful during the model 
training process.
"""
train_data.drop(["Loan_ID"], axis=1, inplace=True)
test_data.drop(["Loan_ID"], axis=1, inplace=True)

# Filling NaN values in the datasets (categorical)
"""
For categorical data, using the mode (most frequent value) to fill missing values is a common approach because it 
maintains the distribution of the data.

Add on:
By filling missing values with the mode, you avoid distorting the data distribution significantly. 
This method is especially useful when the mode is a sensible default or the most likely value for missing entries.

This code fills missing values (NaNs) in specific columns of both the training (train_data) and test (test_data) 
datasets with the most frequent value (mode) in each column.
It creates a list named "fill_mode_cols" that contains the names of the columns for which missing values need to be 
filled using the mode.

Because the mode() func will return a series, I used [0] to assign the first item of the series to the NaN.
The inplace=True parameter ensures that the changes are made directly in the existing DataFrame rather than 
creating a new one.
"""
fill_mode_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
for col in fill_mode_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    test_data[col].fillna(test_data[col].mode()[0], inplace=True)

# Filling NaN values in the datasets (numerical)
fill_median_cols = ["LoanAmount", "Loan_Amount_Term"]
for col in fill_median_cols:
    train_data[col].fillna(train_data[col].median(), inplace=True)
    test_data[col].fillna(test_data[col].median(), inplace=True)

# Creating a new feature 'Total Income'
"""
For this dataset, I will create a new feature called "Total Income" combining the ApplicantIncome and CoapplicantIncome 
because some people might have a low income but strong CoappliantIncome.
"""
train_data["Total Income"] = train_data["ApplicantIncome"] + train_data["CoapplicantIncome"]
test_data["Total Income"] = test_data["ApplicantIncome"] + test_data["CoapplicantIncome"]

# Encoding the target variable
"""
This line converts the categorical values in the Loan_Status column of the train_data DataFrame into numerical values. 
Specifically, it replaces the values "Y" (Yes) and "N" (No) with 1 and 0, respectively.

I am doing this because, most machine learning algorithms require numerical input. By converting the categorical 
labels "Y" and "N" to 1 and 0, the Loan_Status column can be used directly as the target variable for training a 
machine learning model.

It helps in avoiding potential issues that might arise when using algorithms that do not handle categorical variables 
natively. So, I minimize the error I may get in the future, to my knowledge, the error generated here are runtime 
error and they are the hardest to debug.
"""
train_data["Loan_Status"] = train_data["Loan_Status"].replace({"Y": 1, "N": 0})

# Data processing

"""
This line creates a Python list named ordinal_category containing the column names "Dependents" and "Property_Area".

This is done to specifies which columns in the dataset are considered ordinal categorical variables.
"""
ordinal_category = ["Dependents", "Property_Area"]

"""
This line is prepared to transform each categorical column specified in ordinal_category into numerical labels. 
This transformation is crucial because many machine learning algorithms cannot directly handle categorical data and 
require numerical inputs.

I needed to initialize this in other to assess "fit_transform" method in it.
It assigns a unique integer to each category within the column, thus converting categorical data into a 
format suitable for machine learning models.
"""
encoder = LabelEncoder()

for i in ordinal_category:
    train_data[i] = encoder.fit_transform(train_data[i])
    test_data[i] = encoder.fit_transform(test_data[i])  # fit_transform here instead of transform

# One hot encoding using get_dummies() for training and test datasets
"""
This transformation is important because many machine learning algorithms cannot directly process categorical data. 
One-hot encoding ensures that these variables are represented as numerical values (0 or 1), enabling models to 
effectively learn from them.
"""
nominal_category = ['Gender', 'Married', 'Education', 'Self_Employed']
train_data = pd.get_dummies(train_data, columns=nominal_category)
test_data = pd.get_dummies(test_data, columns=nominal_category)

# Ensure the test data has the same columns as the training data
"""
Calculates the set difference, i.e., columns that are in train_data but not in test_data.

Ensures that test_data has the same columns as train_data before further processing or modeling.
"""
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[train_data.columns.drop('Loan_Status')]  # Ensure columns match but drop target variable

# Feature engineering - Copy the original data
"""
This is crucial because data preprocessing steps and modeling can modify datasets, and you may need to refer back 
to the original data for comparison or debugging purposes.
"""
train_data_ML = train_data.copy()
test_data_ML = test_data.copy()

# Remove outliers - Only on numeric columns
"""
To obtain a list of column names from train_data_ML that are numeric types (int64 or float64), excluding the target 
variable 'Loan_Status'.
"""
numeric_cols = train_data_ML.select_dtypes(include=[np.number]).columns.drop('Loan_Status')


"""
The IQR represents the range between the first quartile (Q1) and the third quartile (Q3). It measures the statistical 
dispersion of the data within each numeric column.

The IQR defines the outliers.
"""
Q1 = train_data_ML[numeric_cols].quantile(0.25)
Q3 = train_data_ML[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

"""
This operation identifies and removes rows from train_data_ML where any numeric feature falls outside the acceptable 
range, defined as 1.5 times the Interquartile Range (IQR) from the quartiles (Q1 and Q3).
"""
train_data_ML = train_data_ML[~((train_data_ML[numeric_cols] < (Q1 - 1.5 * IQR)) | (train_data_ML[numeric_cols] >
                                                                                    (Q3 + 1.5 * IQR))).any(axis=1)]

Q1_test = test_data_ML[numeric_cols].quantile(0.25)
Q3_test = test_data_ML[numeric_cols].quantile(0.75)
IQR_test = Q3_test - Q1_test

test_data_ML = test_data_ML[~((test_data_ML[numeric_cols] < (Q1_test - 1.5 * IQR_test)) | (test_data_ML[numeric_cols] > (Q3_test + 1.5 * IQR_test))).any(axis=1)]

# Update the features list to include the new column names generated by get_dummies()

"""
"features" is used to select specific columns from the train_data DataFrame for subsequent 
analysis or modeling tasks related to predicting loan status (Loan_Status). These features are chosen based on domain 
knowledge and data exploration to potentially improve model performance or interpretability.
"""
features = ['Total Income', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
            'Dependents', 'Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes',
            'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes']

label = 'Loan_Status'
X, y = train_data[features].values, train_data[label].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Assume we are working with a loan eligibility dataset where the target variable y has a significant imbalance 
(e.g., 90% of loans are approved, and 10% are not). If we train a model on this imbalanced dataset, it might learn 
to always predict the majority class (loan approved), leading to poor performance on the minority class 
(loan not approved). Using SMOTE to balance the dataset helps ensure that the model receives equal representation from
 both classes during training.
"""
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# Scaling the features
"""
 MinMaxScaler is used to scale both X_train and X_test datasets after splitting them from the original dataset 
 (X and y). This ensures that both training and test data are transformed in the same way, maintaining consistency and 
 preventing data leakage during model training and evaluation.
"""
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model on the training set
"""
the LogisticRegression model is trained on the Iris dataset after splitting it into training (X_train, y_train) and 
test (X_test, y_test) sets. The C=1/0.01 parameter adjusts the regularization strength, and solver="liblinear" 
specifies the optimization algorithm. After training, the model can be used to predict the target variable (y_test) 
for new input data (X_test).
"""
model = LogisticRegression(C=1/0.01, solver="liblinear").fit(X_train, y_train)

"""
This is to get the predicted value.
"""
y_pred = model.predict(X_test)

# Confusion metrics on Logistic Regression
conf_mat = confusion_matrix(y_test, y_pred)
print(f"Confusion Metrics on Logistic Regression:\n{conf_mat}")

print("Overall Precision:", precision_score(y_test, y_pred))
print("Overall Recall:", recall_score(y_test, y_pred))

"""
RandomForestClassifier(n_estimators=1000) initializes a Random Forest classifier with 1000 decision trees. 
After initializing (RF = RandomForestClassifier(n_estimators=1000)), the model can be trained on the training data 
(X_train, y_train) using the fit method (RF.fit(X_train, y_train)). After training, the model can be used to make 
predictions (RF.predict(X_test)) on new data.
"""
RF = RandomForestClassifier(n_estimators=1000)

# Fitting the pipeline to train a random forest model on the training set
"""
This trains the Random Forest model (RF) on the training data (X_train, y_train). After this line executes, model_RF 
will contain the trained model that can be used to make predictions on new data (X_test).
"""
model_RF = RF.fit(X_train, y_train)

"""
Ensure the test data is scaled using the same scaler as the training data.
This is to ensure the model uses features matching the training data
"""
X_test_1 = scaler.transform(test_data[features])

# Make predictions on the test data
predictions = model.predict(X_test_1)

"""
Creating the result DataFrame with the original Loan_ID (the one I kept before the data cleaning) and predictions.
I reserved it to make sure that I can identify individuals with their unique ID later, maybe for reference.
"""
result = pd.DataFrame({"Loan_ID": test_data_ids, "Loan_Status": predictions})
print(result.head())
