import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import sys
import os
import time


# Define parameters
sequence_length = 50
num_hidden_units = 128
num_output_units = 1
batch_size = 32
num_epochs = 10

# Beam search parameters
beam_width = 10

# Load data
precursor_fasta = "RatPrecursormiRNAs.fas"
x_chromosome_fasta = "RatXChromosomeUnmasked.fas"

# Tokenize sequences
tokenizer = Tokenizer(char_level=True)
with open(precursor_fasta) as f:
    data = f.read()
tokenizer.fit_on_texts(data)

# Encode sequences
def load_fasta(filename, as_string=False):
    sequences = []
    with open(filename) as f:
        for line in f:
            if not line.startswith('>'):
                sequence = line.strip()
                sequences.append(tokenizer.texts_to_sequences([sequence])[0])
    if as_string:
        return sum(sequences, [])
    return sequences

precursor_seq = load_fasta(precursor_fasta)
x_chromosome_seq = load_fasta(x_chromosome_fasta, as_string=True)

# Combine X chromosome sequences into a single sequence
x_chromosome_seq = list(np.concatenate(x_chromosome_seq))

# Pad sequences
def pad_sequence(seq, max_len):
    return pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')[0]

input_data = [pad_sequence(seq, sequence_length) for seq in precursor_seq]
x_chromosome_seq = x_chromosome_seq

# Define bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(num_hidden_units, return_sequences=True), input_shape=(sequence_length, 1)))
model.add(Bidirectional(LSTM(num_hidden_units, return_sequences=True)))
model.add(Bidirectional(LSTM(num_hidden_units, return_sequences=True)))
model.add(Bidirectional(LSTM(num_hidden_units)))
model.add(Dense(num_output_units, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prepare data for training
x_train = np.array(input_data).reshape(-1, sequence_length, 1)
y_train = np.ones((len(input_data), 1))

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Train model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs, verbose=1)

# Print training history
for i in range(len(history.history['loss'])):
    print(f"Epoch {i+1}/{num_epochs}")
    print(f"loss: {history.history['loss'][i]:.4f} - accuracy: {history.history['accuracy'][i]:.4f} - val_loss: {history.history['val_loss'][i]:.4f} - val_accuracy: {history.history['val_accuracy'][i]:.4f}")
# Threshold for predicted probabilities
threshold = 0.5

# Generate predictions

predictions = []
for i in range(0, len(x_chromosome_seq) - sequence_length + 1):
    window = x_chromosome_seq[i:i + sequence_length]
    window_np = np.array(window).reshape(1, -1, 1)
    window_prediction = model.predict(window_np, verbose=1)
    if window_prediction > threshold:
        predictions.append((i, window_prediction))

# Identify potential miRNA precursor sequences
predicted_sequences = []
for i, pred in enumerate(predictions):
    if pred > threshold:
        start = i
        end = i + sequence_length
        seq = x_chromosome_seq[start:end]
        decoded_sequence = tokenizer.sequences_to_texts([seq])[0]
        predicted_sequences.append(decoded_sequence)

# Print top predicted miRNA precursor sequences
print("Predicted miRNA precursor sequences:")
for idx, seq in enumerate(predicted_sequences):
    print(f"Sequence {idx + 1}: {seq}")

# Save predicted sequences to a text file
with open("predicted_miRNA_precursors.txt", "w") as output_file:
    for idx, seq in enumerate(predicted_sequences):
        output_file.write(f"Sequence {idx + 1}: {seq}\n")

