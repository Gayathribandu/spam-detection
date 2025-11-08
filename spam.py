import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']


print(data.head())

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

data['cleaned'] = data['message'].apply(preprocess)


print(data.head())

data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})


vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['cleaned'])
y = data['label_num']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n💾 Model and vectorizer saved!")


def predict_spam(message):
    cleaned = preprocess(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"


examples = [
    "Congratulations! You have won a free $1000 Walmart gift card. Click here to claim now.",
    "Hey, are we still meeting at 6 for dinner?"
]

print("\n🔍 Example Predictions:")
for msg in examples:
    print(f"Message: {msg}\nPrediction: {predict_spam(msg)}\n")
