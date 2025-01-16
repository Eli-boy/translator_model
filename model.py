import tkinter as tk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

# Load English and Igbo sentences from files
english_sentences = load_sentences("/Users/mac/Desktop/project/english-igbo-dictionary/english-dictionary.txt")
igbo_sentences = load_sentences("/Users/mac/Desktop/project/english-igbo-dictionary/igbo-dictionary.txt")

# Randomly sample a subset of sentences for training and testing
sample_size = 1000  # Adjust as needed
random.seed(42)  # For reproducibility
sample_indices = random.sample(range(len(english_sentences)), sample_size)
english_sentences_sampled = [english_sentences[i] for i in sample_indices]
igbo_sentences_sampled = [igbo_sentences[i] for i in sample_indices]

# Tokenization
tokenizer_eng = Tokenizer()
tokenizer_eng.fit_on_texts(english_sentences)
eng_sequences = tokenizer_eng.texts_to_sequences(english_sentences)

tokenizer_igbo = Tokenizer()
tokenizer_igbo.fit_on_texts(igbo_sentences)
igbo_sequences = tokenizer_igbo.texts_to_sequences(igbo_sentences)

# Padding sequences
max_seq_length = 50  # Adjust as needed
eng_padded_sequences = pad_sequences(eng_sequences, maxlen=max_seq_length, padding='post')
igbo_padded_sequences = pad_sequences(igbo_sequences, maxlen=max_seq_length, padding='post')

# Splitting into training and validation sets
X_train_eng, X_val_eng, y_train_igbo, y_val_igbo = train_test_split(eng_padded_sequences, igbo_padded_sequences, test_size=0.2, random_state=42)

# Vocabulary sizes
eng_vocab_size = len(tokenizer_eng.word_index) + 1
igbo_vocab_size = len(tokenizer_igbo.word_index) + 1

print("English vocabulary size:", eng_vocab_size)
print("Igbo vocabulary size:", igbo_vocab_size)

# Define the encoder-decoder model
embedding_size = 100
hidden_size = 200

# Encoder
encoder_inputs = tf.keras.layers.Input(shape=(max_seq_length,))
encoder_embedding = tf.keras.layers.Embedding(eng_vocab_size, embedding_size, input_length=max_seq_length)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(hidden_size, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = tf.keras.layers.Input(shape=(max_seq_length,))
decoder_embedding = tf.keras.layers.Embedding(igbo_vocab_size, embedding_size, input_length=max_seq_length)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(igbo_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the model
num_epochs = 20
batch_size = 34
# Store training history after training the model
history = model.fit([X_train_eng, X_train_eng], y_train_igbo,
          validation_data=([X_val_eng, X_val_eng], y_val_igbo),
          batch_size=batch_size, epochs=num_epochs)

# Save the entire model
model.save('my_translation_model.h5')  # Adjust filename as needed

print("Model saved successfully!")


# Extract training and validation loss from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Extract training and validation accuracy lists
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Configure the plot
plt.figure(figsize=(8, 6))  
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()  
plt.title('Loss Curve for Machine Translation Model')
plt.grid(True)  


plt.show()

# Predictions and ground truth labels
y_pred = model.predict([X_train_eng, X_train_eng])
y_true = y_train_igbo

# Convert predictions to class labels
y_pred_class = np.argmax(y_pred, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred_class)

# Print confusion matrix
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Define a function to translate new sentences
def translate(sentence):
    try:
        sentence_sequence = tokenizer_eng.texts_to_sequences([sentence])
        sentence_padded = pad_sequences(sentence_sequence, maxlen=max_seq_length, padding='post')
        predicted_sequence = model.predict([sentence_padded, sentence_padded])
        predicted_sentence = []
        for token in predicted_sequence[0]:
            predicted_word = tokenizer_igbo.index_word[np.argmax(token)]
            if predicted_word == '<end>':
                break
            predicted_sentence.append(predicted_word)
        return ' '.join(predicted_sentence)
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None
