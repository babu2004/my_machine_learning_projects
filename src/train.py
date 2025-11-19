# %% Imports
import numpy as np
import pandas as pd
import pickle


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %% Load dataset
df = pd.read_csv('E:/Machine Learning Zoomcamp/dataset/Salary_Data.csv')

# %% Data preparation
df.columns = df.columns.str.lower().str.replace(' ', '_')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Fix duplicated categories in education_level
df['education_level'] = df['education_level'].str.replace("master's_degree", "master's", regex=False)
df['education_level'] = df['education_level'].str.replace("bachelor's_degree", "bachelor's", regex=False)

# Drop missing values
df = df.dropna()

# Filter salary and job_title
df = df[df['salary'] >= 15000]
valid_titles = df['job_title'].value_counts()[lambda x: x >= 51].index
df = df[df['job_title'].isin(valid_titles)]

# %% Train/Val/Test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=2)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=2)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.salary)
y_val = np.log1p(df_val.salary)
y_test = np.log1p(df_test.salary)

df_train = df_train.drop(columns=['salary'])
df_val = df_val.drop(columns=['salary'])
df_test = df_test.drop(columns=['salary'])

# %% Vectorization
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(df_train.to_dict(orient='records'))
X_val = dv.transform(df_val.to_dict(orient='records'))
X_test = dv.transform(df_test.to_dict(orient='records'))

# %% Model training
rf = RandomForestRegressor(n_estimators=160, random_state=1)
rf.fit(X_train, y_train)

# %% Validation performance
y_pred_val = rf.predict(X_val)
print("Validation RMSE:", mean_squared_error(y_val, y_pred_val))
print("Validation R² score:", r2_score(y_val, y_pred_val))

# %% Test performance
y_pred_test = rf.predict(X_test)
print("Test R² score:", r2_score(y_test, y_pred_test))
print("Test RMSE:", mean_squared_error(y_test, y_pred_test))

# %% Single prediction
id = 469
if id >= len(df_test):
    raise IndexError("Index out of range for df_test")

emp = df_test.iloc[id].to_dict()
X_emp = dv.transform([emp])
y_pred_single = rf.predict(X_emp)

psalary = np.expm1(y_pred_single)[0]
asalary = np.expm1(y_test[id])

print(f"\nActual salary: ₹{int(asalary)}")
print(f"Predicted salary: ₹{int(psalary)}")
print(f"Difference: ₹{int(asalary - psalary)}")

# Convert to USD (approximate)
INR_TO_USD = 88
print(f"\nActual salary: ${int(asalary / INR_TO_USD)}")
print(f"Predicted salary: ${int(psalary / INR_TO_USD)}")
print(f"Difference: ${int((asalary - psalary) / INR_TO_USD)}")

# %% Save model
with open('salary.bin', 'wb') as f_out:
    pickle.dump((dv, rf), f_out)
print("the model is saved to salary.bin!!!")