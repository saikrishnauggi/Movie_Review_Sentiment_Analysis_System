import pandas as pd
import numpy as np
import nltk
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import messagebox

# Load the dataset
df = pd.read_csv(r"e:\Projects\Movie review\IMDB Dataset.csv", encoding='utf-8')
print("Dataset Loaded. Shape:", df.shape)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Encode sentiment labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.2, random_state=42)

# Vectorize text using Bag of Words
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_cv, y_train)

# Predictions
y_pred = nb_model.predict(X_test_cv)

# Evaluation
print("\n--- Accuracy ---")
print("Multinomial Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open('model1.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

with open('bow.pkl', 'wb') as f:
    pickle.dump(cv, f)

print("\nModel and Vectorizer saved as model1.pkl and bow.pkl")

# --- GUI for sentiment prediction ---
def predict_sentiment():
    user_review = review_entry.get("1.0", tk.END).strip()
    if not user_review:
        messagebox.showwarning("Input Error", "Please enter a review.")
        return
    
    cleaned_input = preprocess_text(user_review)
    vector_input = cv.transform([cleaned_input])
    result = nb_model.predict(vector_input)
    sentiment = "Positive" if result[0] == 1 else "Negative"
    
    result_label.config(text=f"Prediction: {sentiment}", fg="green" if sentiment == "Positive" else "red")

# GUI setup
window = tk.Tk()
window.title("Movie Review Sentiment Analyzer")
window.geometry("500x300")
window.config(padx=20, pady=20)

title_label = tk.Label(window, text="Enter a Movie Review", font=("Arial", 14))
title_label.pack(pady=10)

review_entry = tk.Text(window, height=5, width=50, font=("Arial", 12))
review_entry.pack()

predict_button = tk.Button(window, text="Analyze Sentiment", font=("Arial", 12), command=predict_sentiment)
predict_button.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=10)

window.mainloop()
