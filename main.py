import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Dropping unwanted column
train_data.drop(["Loan_ID"], axis=1, inplace=True)
test_data.drop(["Loan_ID"], axis=1, inplace=True)

# Filling NaN values in the datasets (categorical)
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
train_data["Total Income"] = train_data["ApplicantIncome"] + train_data["CoapplicantIncome"]
test_data["Total Income"] = test_data["ApplicantIncome"] + test_data["CoapplicantIncome"]

# Encoding the target variable
train_data["Loan_Status"] = train_data["Loan_Status"].replace({"Y": 1, "N": 0})

# Data processing
ordinal_category = ["Dependents", "Property_Area"]
encoder = LabelEncoder()

for i in ordinal_category:
    train_data[i] = encoder.fit_transform(train_data[i])
    test_data[i] = encoder.transform(test_data[i])

# One hot encoding using get_dummies() for training and test datasets
nominal_category = ['Gender', 'Married', 'Education', 'Self_Employed']
train_data = pd.get_dummies(train_data, columns=nominal_category)
test_data = pd.get_dummies(test_data, columns=nominal_category)

# Ensure the test data has the same columns as the training data
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[train_data.columns.drop('Loan_Status')]  # Ensure columns match but drop target variable

# Feature engineering - Copy the original data
train_data_ML = train_data.copy()
test_data_ML = test_data.copy()

# Remove outliers - Only on numeric columns
numeric_cols = train_data_ML.select_dtypes(include=[np.number]).columns.drop('Loan_Status')

Q1 = train_data_ML[numeric_cols].quantile(0.25)
Q3 = train_data_ML[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

train_data_ML = train_data_ML[~((train_data_ML[numeric_cols] < (Q1 - 1.5 * IQR)) | (train_data_ML[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

Q1_test = test_data_ML[numeric_cols].quantile(0.25)
Q3_test = test_data_ML[numeric_cols].quantile(0.75)
IQR_test = Q3_test - Q1_test

test_data_ML = test_data_ML[~((test_data_ML[numeric_cols] < (Q1_test - 1.5 * IQR_test)) | (test_data_ML[numeric_cols] > (Q3_test + 1.5 * IQR_test))).any(axis=1)]

# Update the features list to include the new column names generated by get_dummies()
features = ['Total Income', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
            'Dependents', 'Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes',
            'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes']

label = 'Loan_Status'
X, y = train_data[features].values, train_data[label].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Assume you are working with a loan eligibility dataset where the target variable y has a significant imbalance 
(e.g., 90% of loans are approved, and 10% are not). If you train a model on this imbalanced dataset, it might learn 
to always predict the majority class (loan approved), leading to poor performance on the minority class 
(loan not approved). Using SMOTE to balance the dataset helps ensure that the model receives equal representation from
 both classes during training.
"""
X, y = SMOTE().fit_resample(X, y)

X = MinMaxScaler().fit_transform(X)

# train a logistic regression model on the training set
model = LogisticRegression(C=1/0.01, solver="liblinear").fit(X_train, y_train)

y_pred = model.predict(X_test)

# print('Accuracy: ', accuracy_score(y_test, y_pred))

print("Overall Precision:", precision_score(y_test, y_pred))
print("Overall Recall:", recall_score(y_test, y_pred))

RF = RandomForestClassifier(n_estimators=1000)

# fit the pipeline to train a random forest model on the training set
model_RF = RF.fit(X_train, y_train)
print(model_RF)

X_test_1 = test_data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Total_Income_log',
                      'Loan_Amount_log', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]

model.fit(X, y)

predictions = model.predict(X_test_1)
result = pd.DataFrame({"Loan_ID": test_data['Loan_ID'], "Loan Status": predictions})
print(result.head())