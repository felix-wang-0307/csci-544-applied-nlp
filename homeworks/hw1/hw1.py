import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup

# ----------------- Data Preprocessing -----------------

# Disable SSL certificate verification warning and download wordnet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('wordnet')

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz
# Data Path: "./amazon_reviews_us_Office_Products_v1_00.tsv"
df = pd.read_csv("./amazon_reviews_us_Office_Products_v1_00.tsv", sep='\t', on_bad_lines='skip')
df = df[['review_body', 'star_rating']]
df = df.dropna()

# Filter out non-integer values
df = df[pd.to_numeric(df['star_rating'], errors='coerce').notnull()]
df['star_rating'] = df['star_rating'].astype(int)

def get_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'


# Print the number of each star rating
print('-' * 40)
print("Number of each star rating:")
print(df['star_rating'].value_counts())
    
# Print the number of positive (>=4), neutral (=3), and negative reviews (<=2)
print('-' * 40)
print("Number of each sentiment:")
print(df['star_rating'].apply(get_sentiment).value_counts())

# Print a sample for each sentiment 
print('-' * 40)
print("Example of a positive review: ", df[df['star_rating'] == 5]['review_body'].iloc[0])
print("Example of a neutral review: ", df[df['star_rating'] == 3]['review_body'].iloc[0])
print("Example of a negative review: ", df[df['star_rating'] == 1]['review_body'].iloc[0])

sample_size = 100000

# set seed for reproducibility
np.random.seed(0)
positive = df[df['star_rating'] > 3].sample(sample_size)
negative = df[df['star_rating'] < 3].sample(sample_size)
df = pd.concat([positive, negative])

def get_sentiment_label(rating):
    if rating >= 4:
        return 1
    else:
        return 0

# replace star ratings with sentiment labels
# 1 - positive, 0 - negative
df['sentiment'] = df['star_rating'].apply(get_sentiment_label)

# remove star_rating column
df = df.drop(columns=['star_rating'])

# ----------------- Data Cleaning -----------------
print("-" * 20, "Data Cleaning", "-" * 20)
print("Samples before cleaning:")
random_indices = np.random.randint(0, len(df), 3)
for i in random_indices:
    print(i, df['review_body'].iloc[i])

# Convert the reviews to lowercase
df['review_body'] = df['review_body'].str.lower()
# Remove HTML tags
df['review_body'] = df['review_body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
# Remove URLs
df['review_body'] = df['review_body'].apply(lambda x: re.sub(r'http\S+', '', x) or re.sub(r'www\S+', '', x))

# perform contractions on the reviews, e.g., won’t → will not. include as many contractions in English that you can think of
# Source: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "i'd": "i had / i would",
    "i'd've": "i would have",
    "i'll": "i shall / i will",
    "i'll've": "i shall have / i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}
# Remove multiple mappings of contractions
for key, value in contractions.items():
    contractions[key] = value.split('/')[0].strip()
# perform contractions on the reviews
df = df.replace(contractions, regex=True)   
# remove non-alphabetical characters
df['review_body'] = df['review_body'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))

print("Samples after cleaning:")
for i in random_indices:
    print(i, df['review_body'].iloc[i])
    
# ----------------- Preprocessing -----------------
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['review_body'] = df['review_body'].apply(remove_stopwords)

# Lemmatize the reviews
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

df['review_body'] = df['review_body'].apply(lemmatize)

print("Samples after preprocessing:")
for i in random_indices:
    print(i, df['review_body'].iloc[i])

# ----------------- Feature Extraction -----------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['review_body']).toarray()
y = df['sentiment']

# print(X.shape)  # (200000, 1000)

# ----------------- Model Training -----------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Perceptron
print("-" * 20, "Perceptron", "-" * 20)
from sklearn.linear_model import Perceptron
model = Perceptron()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print('Training Metrics', '-'*20)
print(classification_report(y_train, y_train_pred))
print('Testing Metrics', '-'*20)
print(classification_report(y_test, y_pred))

# Logistic Regression
print("-" * 20, "Logistic Regression", "-" * 20)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print('Training Metrics', '-'*20)
print(classification_report(y_train, y_train_pred))
print('Testing Metrics', '-'*20)
print(classification_report(y_test, y_pred))

# Naive Bayes
print("-" * 20, "Naive Bayes", "-" * 20)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print('Training Metrics', '-'*20)
print(classification_report(y_train, y_train_pred))
print('Testing Metrics', '-'*20)
print(classification_report(y_test, y_pred))

# SVM
print("-" * 20, "SVM", "-" * 20)
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
