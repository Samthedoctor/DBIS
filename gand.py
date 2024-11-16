from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Course Titles
courses = [
    "Introduction to Python Programming",
    "Introduction to Data Science",
    "Advanced Python Programming",
    "Advanced Machine Learning Techniques",
    "Data Science for Beginners",
    "Python for Data Analysis",
]
df = pd.DataFrame(courses, columns=['title'])

# Vectorize Titles
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
title_vectors = vectorizer.fit_transform(df['title'])

@app.route('/suggest', methods=['POST'])
def suggest_titles():
    user_input = request.args.get('query', '')
    if not user_input.strip():
        return jsonify({"suggestions": []})

    # Vectorize User Input
    input_vector = vectorizer.transform([user_input])

    # Calculate Similarities
    similarity_scores = cosine_similarity(input_vector, title_vectors).flatten()
    df['similarity'] = similarity_scores
    suggestions = df[df['similarity'] > 0.1].sort_values(by='similarity', ascending=False)['title']
    return jsonify({"suggestions": suggestions.tolist()})

if __name__ == '__main__':
    app.run(debug=True,port=8080)