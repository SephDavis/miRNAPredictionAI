import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import sys

# Define parameters
sequence_length = 50
num_hidden_units = 128
num_output_units = 1
batch_size = 32
num_epochs = 10

# Beam search parameters
beam_width = 10

# Load data
precursor_fasta = "precursor_sequences.fas"
x_chromosome_fasta = "x_chromosome_sequence.fa"

# Tokenize sequences
tokenizer = Tokenizer(char_level=True)
with open(precursor_fasta) as f:
    data = f.read()
tokenizer.fit_on_texts(data)

# Encode sequences
def load_fasta(filename):
    with open(filename) as f:
        f.readline()  # Skip header line
        sequence = f.read().replace('\n', '')
    return tokenizer.texts_to_sequences([sequence])[0]

precursor_seq = load_fasta(precursor_fasta)
x_chromosome_seq = load_fasta(x_chromosome_fasta)

# Pad sequences
def pad_sequence(seq, max_len):
    return pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')[0]

input_data = pad_sequence(precursor_seq, sequence_length)
x_chromosome_seq = pad_sequence(x_chromosome_seq, sequence_length)

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
y_train = np.array([1] * len(input_data))

# Train model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs)

# Function to implement Beam search
def beam_search(predictions, beam_width):
    sequences = [[]]
    for prediction in predictions:
        all_candidates = []
        for seq in sequences:
            for idx, value in enumerate(prediction):
                new_seq = seq.copy()
                new_seq.append((idx, value))
                all_candidates.append(new_seq)

        ordered = sorted(all_candidates, key=lambda x: x[-1][1], reverse=True)
        sequences = ordered[:beam_width]

    return [seq[:-1] for seq in sequences]

# Predict new miRNA sequences
predictions = model.predict(np.array(x_chromosome_seq).reshape(-1, sequence_length, 1))
beam_search_results = beam_search(predictions, beam_width)

print("Predicted miRNA sequences:")
for seq in beam_search_results:
    print(tokenizer.sequences_to_texts([seq])[0])
