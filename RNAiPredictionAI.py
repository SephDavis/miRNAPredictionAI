import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Embedding
from keras.utils import pad_sequences, to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import sys
import os
import time
from tqdm import tqdm

# Define parameters
sequence_length = 50
num_hidden_units = 128
num_output_units = 6  # One output unit for each RNA base (A, C, G, U, T, N)
batch_size = 32
num_epochs = 63

# Load data
precursor_fasta = "RatPrecursormiRNAs.fas"
x_chromosome_fasta = "RatXChromosomeUnmasked.fas"
print("Initiating testing...")

# Tokenize sequences
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.word_index = {'a': 1, 'c': 2, 'g': 3, 'u': 4, 't': 5}

with open(precursor_fasta) as f:
    data = [line.strip().upper() for line in f if not line.startswith(">")]
data = [''.join([ch if ch in 'ACGU' else '' for ch in seq]) for seq in data]  # Remove any characters that are not A, C, G, or U, T
tokenizer.fit_on_texts(data)
print("Data loaded successfully.")

# Encode sequences
def load_fasta(filename, as_string=False):
    sequences = []
    current_sequence = ""
    num_chars = 0
    with open(filename) as f:
        for line in tqdm(f, desc="Loading sequences", unit="line"):
            if not line.startswith('>'):
                current_sequence += line.strip().upper()
                num_chars += len(line.strip())
            else:
                if current_sequence:
                    current_sequence = ''.join([ch if ch in 'ACGUT' else '' for ch in current_sequence])  # Remove any characters that are not A, C, G, or U, T
                    encoded_sequence = tokenizer.texts_to_sequences([current_sequence])[0]
                    if as_string:
                        sequences.extend(encoded_sequence)
                    else:
                        sequences.append(encoded_sequence)
                    current_sequence = ""
                    num_chars = 0
        if current_sequence:
            current_sequence = ''.join([ch if ch in 'ACGUT' else '' for ch in current_sequence])  # Remove any characters that are not A, C, G, or U, T
            encoded_sequence = tokenizer.texts_to_sequences([current_sequence])[0]
            if as_string:
                sequences.extend(encoded_sequence)
            else:
                sequences.append(encoded_sequence)
    return sequences

precursor_seq = load_fasta(precursor_fasta)
x_chromosome_seq = load_fasta(x_chromosome_fasta)
print("Sequences Encoded successfully.")

# Pad sequences
def pad_sequence(seq, max_len):
    return pad_sequences([seq], maxlen=max_len, padding='post', truncating='post', value=0)[0]

input_data = [pad_sequence(seq, sequence_length) for seq in precursor_seq]
print("Sequences Padded successfully.")

# Define bidirectional LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=sequence_length))
model.add(Bidirectional(LSTM(num_hidden_units, return_sequences=True)))
model.add(Bidirectional(LSTM(num_hidden_units, return_sequences=True)))
model.add(Bidirectional(LSTM(num_hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(num_output_units, activation='softmax')))
print("LTSM Model Booted successfully.")

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model compiled successfully.")

# Prepare data for training
input_data = [pad_sequence(np.array(seq) - 1, sequence_length) for seq in precursor_seq]

x_train = np.array(input_data)
for seq in input_data:
    try:
        to_categorical(seq, num_classes=num_output_units)
    except IndexError as e:
        print("Error in sequence:", seq)
        raise e

y_train = [to_categorical(seq, num_classes=num_output_units) for seq in input_data]
y_train = np.array(y_train)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Train model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, verbose=1)
print("Training model...")

# Print training history
for i in range(len(history.history['loss'])):
    print(f"Epoch {i + 1}/{num_epochs}")
    print(f"loss: {history.history['loss'][i]:.4f} - accuracy: {history.history['accuracy'][i]:.4f} - val_loss: {history.history['val_loss'][i]:.4f} - val_accuracy: {history.history['val_accuracy'][i]:.4f}")
print("Training completed successfully.")

# Generate predictions
predictions = []
print("Generating predictions...")
for i in range(0, len(x_chromosome_seq) - sequence_length + 1):
    if i % 1000 == 0:  # Print progress for every 1000 steps
        print(f"Progress: {i}/{len(x_chromosome_seq) - sequence_length + 1}")
    window = x_chromosome_seq[i:i + sequence_length]
    window_np = np.array(window).reshape(1, -1, 1)
    window_prediction = model.predict(window_np, verbose=0)
    predictions.append((i, window_prediction))

print("Predictions generated.")

# Identify potential miRNA precursor sequences
predicted_sequences = []
for i, pred in predictions:
    start = i
    end = i + sequence_length
    seq = x_chromosome_seq[start:end]
    decoded_sequence = tokenizer.sequences_to_texts([seq + 1])[0]  # Add 1 back to the sequence
    predicted_sequences.append(decoded_sequence)

# Save predicted sequences to a text file
with open("predicted_miRNA_precursors.txt", "w") as output_file:
    for idx, seq in enumerate(predicted_sequences):
        output_file.write(f"Sequence {idx + 1}: {seq}\n")
        
print("Predicted sequences saved to text file.")

# Check if file is not empty
if os.stat("predicted_miRNA_precursors.txt").st_size == 0:
    print("Error: Text file is empty.")
else:
    print("Text file is not empty.")
