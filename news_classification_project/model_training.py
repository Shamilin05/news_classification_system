import nltk
import pickle
from nltk.corpus import reuters
from sklearn.naive_bayes import MultinomialNB
from vectorization import vectorize_text

nltk.download('reuters')

all_ids = reuters.fileids()

split_idx = int(len(all_ids)*0.8)
train_ids, test_ids = all_ids[:split_idx], all_ids[split_idx:]

def get_text(file_ids):
    return [" ".join(reuters.words(file_id)) for file_id in file_ids]

x_train = get_text(train_ids)
x_test = get_text(test_ids)

y_train = [reuters.categories(file_id)[0] for file_id in train_ids]
y_test = [reuters.categories(file_id)[0] for file_id in test_ids]

x_train_vec = vectorize_text(x_train)

model = MultinomialNB()
model.fit(x_train_vec, y_train)

with open("vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)
x_test_vec = vectorizer.transform(x_test)

y_pred = model.predict(x_test_vec)

with open("model.pkl","wb") as model_file:
    pickle.dump(model, model_file)

print("Training complete! Model saved")
