**Employee Salary Prediction using Machine Learning
Overview**

This project predicts the salary of an employee based on their experience, age, education level, and job title.
I chose this project out of curiosity—when meeting a new person, I often wonder how much they might be earning based on their background, and this model helps quantify that intuition using data.

The project uses a Random Forest Regressor trained on a Kaggle dataset and includes a complete machine learning pipeline with preprocessing, model training, evaluation, and deployment support using Flask, Pipenv, Docker, and Gunicorn.

**Dataset**

Source: Kaggle - Salary_Data
Rows: 6702
Columns: 6

| Column Name         | Description                     |
| ------------------- | ------------------------------- |
| Age                 | Age of the employee             |
| Gender              | Male/Female                     |
| Education Level     | Bachelor's, Master's, PhD, etc. |
| Job Title           | Employee role                   |
| Years of Experience | Total work experience           |
| Salary              | Annual salary (target variable) |


**Technologies Used**

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
Pipenv
Docker
Flask
Gunicorn
