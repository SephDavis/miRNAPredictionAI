import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Define parameters
sequence_length = 50
num_hidden_units = 128
num_output_units = 1
batch_size = 32
num_epochs = 10

def seq_to_one_hot(sequence):
    one_hot = []
    for char in sequence:
        vec = [0, 0, 0, 0]
        if char == 'A':
            vec[0] = 1
        elif char == 'C':
            vec[1] = 1
        elif char == 'G':
            vec[2] = 1
        elif char == 'T':
            vec[3] = 1
        one_hot.append(vec)
    return one_hot

def load_multiple_fasta(filenames, labels):
    seqs_and_one_hot = []
    for idx, filename in enumerate(filenames):
        with open(filename) as f:
            f.readline()  # Skip header line
            sequence = f.read().replace('\n', '')
        one_hot_seq = seq_to_one_hot(sequence)
        seqs_and_one_hot.append((sequence, one_hot_seq, labels[idx]))
    return seqs_and_one_hot

def predict_rnai_sequences(model, new_seq_file):
    _, new_seq_one_hot, _ = load_fasta(new_seq_file, None)
    predictions = model.predict(np.reshape(new_seq_one_hot[:-(len(new_seq_one_hot) % sequence_length)], (-1, sequence_length, 4)))
    return predictions

# Load siRNA, shRNA, and DNA sequences
siRNA_file = "siRNA_example.fasta"
shRNA_file = "shRNA_example.fasta"
DNA_file = "DNA_example.fasta"
seqs_and_one_hot = load_multiple_fasta([siRNA_file, shRNA_file, DNA_file], [1, 1, 0])

# Create input/output data from one-hot encoded sequences
input_data = []
output_data = []
for seq, one_hot_seq, label in seqs_and_one_hot:
    for i in range(len(seq) - sequence_length - 1):
        input_data.append(one_hot_seq[i:i+sequence_length])
        output_data.append(label)
input_data = np.reshape(input_data, (len(input_data), sequence_length, 4))
output_data = np.array(output_data)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(num_hidden_units, return_sequences=True, input_shape=(sequence_length, 4)))
model.add(LSTM(num_hidden_units, return_sequences=True))
model.add(LSTM(num_hidden_units, return_sequences=True))
model.add(LSTM(num_hidden_units))
model.add(Dense(num_output_units, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs)

# Predict new siRNA and shRNA molecules
new_seq_file = "new_sequence.fasta"
predictions = predict_rnai_sequences
(model, new_seq_file)
predicted_rnai_sequences = []
threshold = 0.5

for i, prediction in enumerate(predictions):
if prediction >= threshold:
start_idx = i * sequence_length
end_idx = (i + 1) * sequence_length
predicted_rnai_sequences.append(new_seq_file[start_idx:end_idx])

print("Predicted RNAi sequences:")
for seq in predicted_rnai_sequences:
print(seq)
