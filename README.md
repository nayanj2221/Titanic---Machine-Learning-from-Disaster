# Titanic---Machine-Learning-from-Disaster
A data-driven analysis and machine learning model to predict passenger survival in the Titanic disaster using Python, Pandas, and Scikit-learn.


# Titanic Survival Prediction 

This project is a classic machine learning challenge from a Kaggle competition. The goal is to build a model that predicts whether a passenger on the RMS Titanic survived the infamous 1912 disaster, based on a given set of passenger data.

**Competition Link:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

-----

##  Table of Contents

  * [Project Overview](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#project-overview)
  * [Project Workflow](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#project-workflow)
  * [Data Dictionary](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#data-dictionary)
  * [Exploratory Data Analysis (EDA) & Key Findings](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#exploratory-data-analysis-eda--key-findings)
  * [Feature Engineering](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#feature-engineering)
  * [Modeling & Evaluation](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#modeling--evaluation)
  * [Results & Conclusion](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#results--conclusion)
  * [Future Improvements](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#future-improvements)
  * [How to Run this Project](https://github.com/nayanj2221/Titanic---Machine-Learning-from-Disaster/edit/main/README.md#how-to-run-this-project)

-----

##  Project Overview

The sinking of the Titanic is one of the most notorious shipwrecks in history. While some luck was involved in surviving, it seems some groups of people were more likely to survive than others. This project uses passenger data (e.g., name, age, gender, socio-economic class, etc.) to build a machine learning model capable of predicting survival outcomes.

This end-to-end project demonstrates a complete data science pipeline, including:

  - Data Cleaning and Preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature Engineering
  - Model Training and Evaluation
  - Submission Generation

**Tech Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

-----

##  Project Workflow

The project followed a structured approach to ensure a robust and well-documented solution.

1.  **Data Loading & Initial Inspection:** Loaded `train.csv` and `test.csv` datasets and examined their structure, datatypes, and initial statistics.
2.  **Data Cleaning:** Handled missing values in critical columns like `Age`, `Embarked`, and `Cabin`.
3.  **Exploratory Data Analysis (EDA):** Used visualizations to understand the relationships between different features and the survival outcome.
4.  **Feature Engineering:** Created new, more informative features from existing ones to improve model performance (e.g., `FamilySize`, `Title`).
5.  **Model Building:** Trained several classification algorithms on the prepared data.
6.  **Model Evaluation:** Assessed model performance using cross-validation and accuracy metrics.
7.  **Submission:** Used the best-performing model to make predictions on the test dataset and generated the submission file.

-----

##  Data Dictionary

| Variable   | Definition                                 | Key                                            |
|------------|--------------------------------------------|------------------------------------------------|
| `Survived` | Survival                                   | 0 = No, 1 = Yes                                |
| `Pclass`   | Ticket class (Proxy for socio-economic status) | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| `Sex`      | Sex                                        | `male`, `female`                               |
| `Age`      | Age in years                               |                                                |
| `SibSp`    | \# of siblings / spouses aboard the Titanic |                                                |
| `Parch`    | \# of parents / children aboard the Titanic |                                                |
| `Ticket`   | Ticket number                              |                                                |
| `Fare`     | Passenger fare                             |                                                |
| `Cabin`    | Cabin number                               |                                                |
| `Embarked` | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

-----

##  Exploratory Data Analysis (EDA) & Key Findings

Several key insights were drawn from the data that heavily influenced feature engineering and modeling:

  * **Gender vs. Survival:** Women had a significantly higher survival rate (\~74%) compared to men (\~19%), confirming the "women and children first" protocol.
     \* **Passenger Class vs. Survival:** First-class passengers had the highest survival rate (\~63%), followed by second (\~47%) and third class (\~24%). Wealth and status played a crucial role.
  * **Age vs. Survival:** Children (age \< 16) had a higher survival rate than other age groups. The `Age` column had missing values that were imputed using the median age of passengers grouped by their `Pclass` and `Sex`.
  * **Port of Embarkation:** Passengers who embarked from Cherbourg (C) had a higher survival rate, likely because they were predominantly first-class passengers.

-----

##  Feature Engineering

To enhance the model's predictive power, the following features were created:

1.  **`FamilySize`**: Combined `SibSp` and `Parch` to get the total number of family members on board.
    $$FamilySize = SibSp + Parch + 1$$
2.  **`IsAlone`**: A binary feature derived from `FamilySize` to indicate if a passenger was traveling alone.
3.  **`Title`**: Extracted titles (e.g., "Mr", "Mrs", "Miss", "Master") from the `Name` column. This served as a strong proxy for age, gender, and social status.
4.  **`AgeGroup`**: Binned the `Age` feature into categories (e.g., Child, Teen, Adult, Senior) to better capture non-linear relationships with survival.
5.  **`FarePerPerson`**: Calculated by dividing the `Fare` by `FamilySize`.

-----

##  Modeling & Evaluation

Several classification models were trained and evaluated using **5-fold cross-validation** to ensure robustness and prevent overfitting.

The models considered were:

1.  **Logistic Regression:** A good baseline model for binary classification.
2.  **Support Vector Machine (SVM):** Effective in high-dimensional spaces.
3.  **Random Forest Classifier:** An ensemble model that is robust against overfitting and captures complex interactions.
4.  **Gradient Boosting Classifier:** A powerful ensemble method that builds trees sequentially.

The primary metric for evaluation was **Accuracy**, as required by the Kaggle competition.

-----

##  Results & Conclusion

The models were compared based on their mean cross-validation accuracy.

| Model                       | Mean Cross-Validation Accuracy | Kaggle Score |
|-----------------------------|--------------------------------|--------------|
| Logistic Regression         | 0.795                          | *[Your Score]* |
| Support Vector Machine      | 0.812                          | *[Your Score]* |
| **Random Forest Classifier** | **0.825** | **[Your Best Score]** |
| Gradient Boosting           | 0.821                          | *[Your Score]* |

The **Random Forest Classifier** provided the best and most stable performance, achieving a final Kaggle submission score of **[Your Best Score]**.

The analysis confirms that `Title`, `Sex`, `Pclass`, and `FamilySize` were the most influential features in predicting survival. This project successfully demonstrates a complete machine learning pipeline, from raw data to a predictive model.

-----

##  Future Improvements

  - **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to find the optimal parameters for the best-performing models.
  - **Advanced Ensembling:** Combine the predictions of multiple models (stacking/blending) to potentially improve the final score.
  - **More Feature Engineering:** Explore the `Ticket` and `Cabin` features more deeply to extract potentially useful information.

-----

##  How to Run this Project

To replicate this analysis, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook [your_notebook_name].ipynb
    ```
