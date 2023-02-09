from joblib import load
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

print('Importing dataset from ./data/...')

# Read csv into pandas DataFrame
df = pd.read_csv(r'./data/spam_ham_dataset.csv')
# Removing Unnecessary column
df.drop({'Unnamed: 0'}, axis=1, inplace = True)

# Downsampling ham to balance the dataset
ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']
ham = ham.sample(n = len(spam), random_state=123)
balanced_data = ham.append(spam).reset_index(drop = True)

# Removing Unnecessary column
balanced_data.drop({'label'}, axis=1, inplace = True)

X = balanced_data['text']
stemmer = PorterStemmer()
X = X.apply(stemmer.stem)
vectorizer = CountVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

model = input('Which algorithm is to be used? (random forest / multinomial nb)')

if model.lower() == 'random forest':
    print(f'Loading {model.lower()} model...')
    rf = load('./src/models/randomforest.joblib')
    y_pred = rf.predict(X_vect)
    y_pred_df = pd.DataFrame(data=y_pred, columns=["predicted_label_num"])
    results = balanced_data.join(y_pred_df)
    results.to_csv(r'./data/rf_results.csv')
    print('Results saved into csv format!')

elif model.lower() == 'multinomial nb':
    print(f'Loading {model.lower()} model...')
    nb = load('./src/models/multinomialnb.joblib')
    y_pred = nb.predict(X)
    y_pred_df = pd.DataFrame(data=y_pred, columns=["predicted_label_num"])
    results = balanced_data.join(y_pred_df)
    results.to_csv(r'./data/multinomialnb_results.csv')
    print('Results saved into csv format!')

else:
    print('Please select a valid option. (random forest / multinomial nb)')