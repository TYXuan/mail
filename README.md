# Spam Mails Detection

By Tan Yu Xuan

email: yxtan05@gmail.com

## Project Organisation


    ├── run.sh            <- .sh file which preprocesses the raw data and performs predictions.
    |
    ├── README.md          <- The top-level README for users of this project.
    |
    ├── data               <- Where spam_ham_dataset.csv should be placed in & processed results are stored in.
    | 
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── eda.ipynb          <- Exploratory data analysis.
    |
    ├── src                <- Source code for use in this project.
    |   ├── __init__.py    <- Makes src a Python module
    |   |
    |   |── modelling.py   <- Script to perform predictions based on processed dataset.
    |   |
    |   └── models         <- Dumps of trained models
    |         ├── randomforest.joblib
    |         |
    |         └── multinomialnb.joblib
    |
    └── algorithm_used.txt <- Input the algorithm to be used (random forest / multinomial nb).
--------

# Instructions

## 1. Setting up the enviroment

### Install the required libraries from your terminal

- In your command shell/terminal locate the directory where requirements.txt is located and type in the following:

      conda create --name mail --file requirements.txt

### Once all packages/libraries are installed activate the installed enviroment

- Key in the following code into your shell:

      conda activate mail

## 2. Placing your data

- Place the data into the `data` folder of this directory

## 3. Running the model

- In your command shell/terminal move to where run.sh is located and key in the following command
    
      ./run.sh
--------

# Engineering Pipeline

1. Import dataset
2. Checking for duplicates & missing values
3. Correcting abnormal data
4. Build additional features 
5. Removing punctuations, converting text data into lower case form, tokenizing, remove stopwords and stemming words.
6. Saving processed dataset out as .csv
7. fit model & pickle it out
8. Predict with features & unpickled model
9. Output results in .csv file

# EDA Overview

The EDA contains overview of key findings from the dataset. Choices made in the pipeline are based on those findings, particularly any feature engineerings for modelling. Please keep the details of the EDA in the `.ipynb` which serves as a quick summary.

# Model choice

The following classification models are chosen for this project:
 
  1. Random Forest Classifier
  2. Multinomial NB

  # Model Evaluation

For classification models, the baseline was set to an accuracy of 0.5 where the model has a 50:50 chance of a correct prediction.
The goal is for our models to perform better than a coin toss.

In addition, the purpose is precisely identify 'spam' mails and also reduce the number of normal accounts which is labelled as 'spam'. 

Therefore, it is essential to maximize the precision of `'label_num'=1` (`spam`) and recall of `'label_num'=0` (`ham`).

### Scoring metrics
**ROC AUC scores**

| model                    | training set | test set|
|--------------------------|--------------|---------|
|Multinomial NB            |    0.99     |   `0.97` |
|`Random Forest Classifier`|      1      |   `0.97` |

From the ROC AUC scores, both Multinomial NB and Random Forest Classifier perform equally well. 

Based on the test set, we see that the classifiers have a 97% probability of identifying the correct 'label_num' which is better than our baseline.

Comparing the scores from the training set and test set, there might be some overfitting as the models perform slightly worse on the unseen data.

### Multinomial NB
#### a. Confusion Matrix
|             | label_num=0 | label_num=1 |
|-------------|-------------|-------------|
| label_num=0 |     `428`   |     `11`    | 
| label_num=1 |      19     |    `420`    |

#### b. Classification Report

|     label   | precision | recall | f1-score|
|-------------|-----------|--------|---------|
| label_num=0 |   0.96    | `0.97` |   0.97  |
| label_num=1 |  `0.97`   |  0.96  |   0.97  |

### Random Forest Classifer
#### a. Confusion Matrix
|             | label_num=0 | label_num=1 |
|-------------|-------------|-------------|
| label_num=0 |     `416`   |     `23`    | 
| label_num=1 |      4      |    `435`    |

#### b. Classification Report

|     label   | precision | recall | f1-score|
|-------------|-----------|--------|---------|
| label_num=0 |   0.99    | `0.95` |   0.97  |
| label_num=1 |  `0.95`   |  0.99  |   0.97  |

Based on the classification report, `Multinomial NB` has the highest `precision value of 97%` for `label_num=1` and `recall value of 97%` for `label_num=0`.
It slightly outperforms the Random Forest Classifer.

# Conclusion
Comparing the types of models, MultinomialNB is definitely the most accurate in identifying mail types (spam or ham).

