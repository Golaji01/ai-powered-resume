import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv("resume_dataset.csv")

# Encode job domain
le = LabelEncoder()
df["domain_encoded"] = le.fit_transform(df["domain"])

# TF-IDF vectorization on skills
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["skills"])
y = df["domain_encoded"]

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Save the model, vectorizer, and label encoder
with open("resume_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model training completed and saved.")
