from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = ["what is ai", "what is python"]
answers = ["AI means Artificial Intelligence", "Python is a programming language"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

while True:
    user = input("You: ")

    if user == "exit":
        break

    user_vec = vectorizer.transform([user])
    result = cosine_similarity(user_vec, X)
    index = result.argmax()

    print("Bot:", answers[index])