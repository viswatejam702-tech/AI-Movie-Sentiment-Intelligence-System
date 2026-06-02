import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv(r"C:\ds and AI\ALL DATASETS\finalReviews.csv")

X = df["review"]
y = df["label"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=10000
)

X_train = tfidf.fit_transform(X_train)

model = MultinomialNB()

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "models/mnb_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf.pkl"
)

print("Training Complete")