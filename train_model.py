import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv("emails.csv")
data.columns = data.columns.str.strip()
data = data.rename(columns={"Category": "label", "Message": "message"})
data.dropna(subset=["message", "label"], inplace=True)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save files
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
