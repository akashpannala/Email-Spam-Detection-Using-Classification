import numpy as np
import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# Load the dataset
df_all = pd.read_csv('spam_ham_dataset.csv')

# Data preprocessing
df_all.drop(columns=["Unnamed: 0","label"], axis=1, inplace=True)

def preprocessing(text):
    """Enhanced text preprocessing function"""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in stopwords.words('english') and w not in string.punctuation]
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)

# Apply preprocessing
print("Preprocessing text data...")
df_all["text"] = df_all['text'].apply(preprocessing)

# TF-IDF Vectorization
print("Vectorizing text data...")
tf = TfidfVectorizer(max_features=5000)  # Limit features to prevent overfitting
x = tf.fit_transform(df_all['text']).toarray()
y = df_all['label_num']

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply scaling
print("Applying feature scaling...")
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

# Train model with scaled data
print("Training model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(xtrain_scaled, ytrain)

# Predictions and evaluation
ypred = model.predict(xtest_scaled)

acc = accuracy_score(ytest, ypred)
pre = precision_score(ytest, ypred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f}")

# Save the model and preprocessing objects
print("Saving model and preprocessing objects...")
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(tf, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')

# Also save using pickle for compatibility
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tf, f)

with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and preprocessing objects saved successfully!")