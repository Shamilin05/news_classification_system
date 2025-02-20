import pickle
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import preprocess

def vectorize_text(texts):
    processed_text = [preprocess(text) for text in texts]
    vectorizer = CountVectorizer(max_features = 5000)
    x = vectorizer.fit_transform(processed_text)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return x
print("Vectorization complete!! Vectorizer saved")
