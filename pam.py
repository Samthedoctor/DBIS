import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
data = {
    'title': [
"Introduction to Machine Learning",
"Advanced Machine Learning Techniques",
"Machine Learning for Computer Vision",
"Introduction to Data Science",
"Advanced Data Science and Analytics",
"Data Science for Business Analytics",
"Introduction to Python Programming",
"Advanced Python Development",
"Python for Data Analysis",
"Web Development Fundamentals",
"Advanced Web Development",
"Full Stack Web Development",
"Digital Marketing Essentials",
"Advanced Digital Marketing Strategies",
"Social Media Marketing Mastery",
"Introduction to Artificial Intelligence",
"Advanced AI and Deep Learning",
"AI for Business Applications",
"Cloud Computing Fundamentals",
"AWS Solutions Architect",
"Microsoft Azure Development",
"Cybersecurity Fundamentals",
"Advanced Cybersecurity",
"Network Security Essentials",
"UI/UX Design Basics",
"Advanced UI/UX Design",
"Mobile UI Design",
"Business Analytics Fundamentals",
"Advanced Business Analytics",
"Predictive Analytics for Business",
"Financial Management Basics",
"Investment Strategies",
"Personal Finance Planning",
"Project Management Essentials",
"Agile Project Management",
"PMP Certification Preparation",
"Graphic Design Fundamentals",
"Advanced Graphic Design",
"Brand Identity Design",
"Content Marketing Basics",
"Advanced Content Strategy",
"SEO Content Writing",
"Mobile App Development Basics",
"iOS App Development",
"Android App Development",
"DevOps Fundamentals",
"Advanced DevOps Practices",
"Container Orchestration",
"Data Analytics Fundamentals",
"Advanced Data Analytics",
"Big Data Analytics",
"Blockchain Fundamentals",
"Smart Contract Development",
"DeFi Development",
"Product Management Basics",
"Advanced Product Management",
"Product Analytics",
"Motion Graphics Basics",
"Advanced Motion Graphics",
"3D Animation",
"Database Management Basics",
"Advanced SQL Development",
"NoSQL Database Architecture",
"Healthcare Analytics Foundation",
"Clinical Data Management",
"Medical Informatics Essentials",
"Game Development Principles",
"Unity Game Programming",
"Game Design Theory",
"Natural Language Processing",
"Speech Recognition Systems",
"Text Mining and Analytics",
"Quantum Computing Basics",
"Quantum Algorithms",
"Quantum Machine Learning",
"Supply Chain Analytics",
"Logistics Management Systems",
"Inventory Optimization",
"Renewable Energy Technologies",
"Solar Power Engineering",
"Wind Energy Systems",
"Music Production Fundamentals",
"Audio Engineering",
"Digital Music Composition",
"Robotics Engineering Basics",
"Industrial Automation Systems",
"Robot Motion Planning",
"Virtual Reality Development",
"Augmented Reality Applications",
"Mixed Reality Design",
"Network Infrastructure Design",
"5G Technology Implementation",
"Network Security Architecture",
"Emotional Intelligence Leadership",
"Strategic Management Skills",
"Change Management Principles",
"Sustainable Architecture Design",
"Urban Planning Fundamentals",
"Landscape Architecture",
"Interior Design Principles",
"Commercial Space Planning",
"Sustainable Interior Design",
"Food Science Technology",
"Nutrition and Dietetics",
"Food Safety Management",
"Environmental Science Fundamentals",
"Climate Change Studies",
"Conservation Biology",
"Digital Photography Basics",
"Photo Post-Processing",
"Commercial Photography",
"Video Production Essentials",
"Video Editing Techniques",
"Documentary Filmmaking",
"Creative Writing Workshop",
"Screenwriting Fundamentals",
"Technical Writing Skills",
"Data Visualization Design",
"Interactive Dashboard Creation",
"Visual Analytics Tools",
"Bioinformatics Analysis",
"Genomic Data Science",
"Computational Biology",
"Financial Risk Management",
"Algorithmic Trading Strategies",
"Cryptocurrency Trading",
"Sports Science Foundation",
"Athletic Performance Analysis",
"Sports Nutrition Planning",
"Construction Project Management",
"Building Information Modeling",
"Construction Cost Estimation",
"Digital Forensics Essentials",
"Malware Analysis Techniques",
"Incident Response Planning",
"Manufacturing Process Design",
"Quality Control Systems",
"Lean Manufacturing Principles",
"Chemical Process Engineering",
"Petroleum Engineering Basics",
"Polymer Science Technology",
"Public Relations Strategy",
"Crisis Communication Management",
"Corporate Communications",
"Search Engine Optimization",
"Email Marketing Campaigns",
"Influencer Marketing Strategy",
"Human Resource Management",
"Talent Acquisition Strategy",
"Employee Relations Management"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Prepare Data
def prepare_data(titles):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in titles:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_length = max(len(x) for x in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    # Create predictors and label
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = np.eye(total_words)[y]  # One-hot encoding

    return X, y, total_words, tokenizer

X, y, total_words, tokenizer = prepare_data(df['title'].values)

# Build the LSTM model
def create_model(total_words):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=X.shape[1]))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model(total_words)

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Predict the title based on input
def predict_title(input_text, model, tokenizer, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=-1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

# Example usage
input_text = "Search Engine"
predicted_word = predict_title(input_text, model, tokenizer, X.shape[1])
print(f"Predicted next word: {predicted_word}")