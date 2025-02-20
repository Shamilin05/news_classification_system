import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
def preprocess(text):
    text = text.lower()
    words = text.split()
    words = [word for word in words if word.isalpha() and word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    cleaned_text = " ".join(words)
    return cleaned_text

