# Employee Salary Prediction using Machine Learning
## Overview

* This project predicts the salary of an employee based on their experience, age, education level, and job title.
I chose this project out of curiosity—when meeting a new person, I often wonder how much they might be earning based on their background, also i feel it awkward to ask them how much you earn and this model helps quantify that intuition using data. and also it help to know the market value of their job role.

* The project uses a Random Forest Regressor trained on a Kaggle dataset and includes a complete machine learning pipeline with preprocessing, model training, evaluation, and deployment support using Flask, Pipenv, Docker, and Gunicorn.

## Dataset

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
| Salary              | monthly salary in indian rupees (target variable) |


## Technologies Used

  * Python
  * Pandas
  * NumPy
  * Matplotlib
  * Seaborn
  * Scikit-learn
  * Jupyter Notebook
  * Pipenv
  * Docker
  * Flask
  * Gunicorn
 
## Preprocessing

The following preprocessing steps were applied:

 * Label encoding for categorical columns: Gender, Education Level, Job Title
 * Handling missing values
 * Train-test split
 * Histplot visualization for salary distribution

## Results

The final model achieved:

R² Score: 0.9821614849153605

This indicates excellent predictive performance and strong generalization on unseen data.

## Setup
### Local Setup
**Clone the Repository:**
'git clone https://github.com/ babu2004/my_machine_learning_projects.git
cd my_machine_learning_projects'
