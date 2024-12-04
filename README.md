#Toxic comment classification is the process of identifying and categorizing harmful or offensive content in text, commonly used in online platforms to ensure safe and respectful communication. This task involves detecting various types of toxic behavior, such as obscene language, insults, threats, hate speech, harassment, and spam. By leveraging natural language processing techniques, platforms can automatically flag or remove inappropriate content to improve user experiences and promote healthier online interactions. Approaches to toxic comment classification range from rule-based systems, which rely on predefined keywords or patterns, to advanced machine learning models, including traditional algorithms and deep learning techniques like transformers (e.g., BERT). This technology plays a critical role in content moderation, policy enforcement, and social media analytics, helping platforms maintain compliance with community standards and create inclusive digital environments.
# Toxic-comment-classification
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from gensim.models import FastText
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
train_df = pd.read_csv('path_to_your_dataset.csv')

# Display the first few rows
print(train_df.head())

# Load spaCy model for text processing
nlp = spacy.load('en_core_web_sm')

# Function to clean the text data
def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Clean the comments
train_df['cleaned_text'] = train_df['comment_text'].apply(clean_text)

# Split the dataset into training and testing sets
comment_texts = train_df['cleaned_text']
target_labels = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

X_train, X_test, y_train, y_test = train_test_split(comment_texts, target_labels, test_size=0.2, random_state=42)

# Create FastText embeddings
w2v_model = FastText(sentences=X_train.apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, workers=4)

# Function to get embeddings for comments
def get_embedding(model, comment):
    words = comment.split()
    embeddings = [model.wv[word] for word in words if word in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

# Generate embeddings for training and testing data
X_train_word_emb = np.array([get_embedding(w2v_model, comment) for comment in X_train])
X_test_word_emb = np.array([get_embedding(w2v_model, comment) for comment in X_test])

# Build the neural network model
model = keras.Sequential([
    layers.Input(shape=(X_train_word_emb.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='sigmoid')  # 6 output classes for multi-label classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_word_emb, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
predictions = model.predict(X_test_word_emb)
predictions = (predictions > 0.5).astype(int)

# Calculate F1 scores for each label
f1_scores = []
for i in range(6):
    f1 = f1_score(y_test.iloc[:, i], predictions[:, i], average='micro')
    f1_scores.append(f1)
    print(f"F1 score for label {y_train.columns[i]}: {f1}")

print("Average F1 Score:", np.mean(f1_scores))

# Example comments for classification
comments_to_classify = [
    "You are an idiot!",
    "This product is fantastic!",
    "I hope you get what you deserve.",
    "You're doing great!"
]

# Predict toxicity for new comments
predictions_new = model.predict(np.array([get_embedding(w2v_model, comment) for comment in comments_to_classify]))
predictions_new = (predictions_new > 0.5).astype(int)

# Display predictions
for comment, pred in zip(comments_to_classify, predictions_new):
    print(f"Comment: '{comment}' | Toxicity: {pred}")
