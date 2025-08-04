# clean-prep-ML
A powerfull toolkit for data cleaning and preprocessing in machine learning. Streamline your workflow with ready-to-use functions for handling missing values, encoding, scaling, outlier detection, and more. Turn messy data into clean, ML-ready datasets in minutes!         
Data Cleaning and Preprocessing:
# Data Cleaning and Preprocessing for Machine Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Load the dataset (using sample data here)
data = {
    'Age': [25, 30, np.nan, 35, 28, 40, 29, 31],
    'Salary': [50000, 54000, 58000, np.nan, 52000, 62000, 61000, 59000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', np.nan, 'Female', 'Male'],
    'Department': ['Sales', 'IT', 'HR', 'IT', 'Sales', 'HR', 'HR', 'IT'],
    'Purchase': [1, 0, 1, 1, 0, 0, 1, 1]  # Target variable
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Handle Missing Values
# Fill numerical columns with median
for col in ['Age', 'Salary']:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
for col in ['Gender']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 3. Convert categorical features to numerical
# One-hot encode Department
df = pd.get_dummies(df, columns=['Department'], prefix='Dept')

# Label encode Gender
gender_map = {'Male': 0, 'Female': 1}
df['Gender'] = df['Gender'].map(gender_map)

# 4. Normalize/Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Salary']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 5. Detect and handle outliers
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df.boxplot(column=['Age'])
plt.title('Age Boxplot')

plt.subplot(1, 2, 2)
df.boxplot(column=['Salary'])
plt.title('Salary Boxplot')
plt.tight_layout()
plt.show()

# Remove outliers using IQR method
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)].copy()

for col in numerical_cols:
    df = remove_outliers(df, col)

print("\nCleaned Dataset:")
print(df)
print("\nData Types After Cleaning:")
print(df.dtypes)

# Final preprocessed data is now ready for ML
X = df.drop('Purchase', axis=1)
y = df['Purchase']                                                                                                                                                                                                                                                                 
Dataset Description:
   The goal is to streamline workflows by providing ready-to-use functions for various tasks such as:

=> Handling missing values

=> Encoding categorical data

=> Scaling features

=> Detecting outliers

=> Transforming messy data into clean, machine learning-ready datasets                                                                                                                                                                                                               For the data cleaning and preprocessing task, I performed the following steps:

=> Data Loading: Imported the raw dataset into a Pandas DataFrame for analysis.

=> Handling Missing Values: Identified and addressed missing values by filling numerical columns with their median and categorical columns with their mode to ensure data completeness.

=> Data Transformation: Encoding Categorical Variables: Used one-hot encoding for categorical features to convert them into a numerical format suitable for machine learning models.

=> Feature Scaling: Applied standardization to numerical features to ensure they have a mean of 0 and a standard deviation of 1, which helps improve model performance.

=> Outlier Detection and Removal: Analyzed the data for outliers using box plots and removed them based on the Interquartile Range (IQR) method to enhance data quality.

=> Data Export: Saved the cleaned and preprocessed data to a new CSV file for further analysis or model training.

=> Documentation: Created a structured repository with clear documentation, including a README file, to explain the project, its structure, and how to use the code.

This comprehensive approach ensured that the dataset was clean, well-structured, and ready for machine learning applications.
print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)
