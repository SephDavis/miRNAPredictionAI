
Here's a breakdown of how the program works:

1. The `seq_to_one_hot` function converts a DNA sequence (string) into a one-hot encoded list of lists, 
    where each nucleotide is represented as a list of length 4 (one-hot encoding).
    
2. The `load_multiple_fasta` function loads multiple FASTA files and returns a list of tuples, each containing the raw sequence, 
    one-hot encoded sequence, and the label (siRNA or shRNA = 1, DNA = 0).
    
3. The `predict_rnai_sequences` function takes a trained model and a new FASTA file as input. 
    It uses the model to predict the probability of each sequence in the file inducing RNAi. The function returns a list of predictions.
    
4. The input and output data are prepared from the one-hot encoded sequences and their corresponding labels. 
    The data is split into training and validation sets using `train_test_split`.
    
5. An LSTM model with four LSTM layers is defined and compiled. The model is then trained on the training data.

6. The `predict_rnai_sequences` function is called with the trained model and a new FASTA file to predict siRNA and shRNA molecules that may induce RNAi.

7. Predicted RNAi sequences are printed if their prediction value is greater than or equal to the defined threshold (0.5 in this case).

With this code, you can now predict RNAi sequences from new FASTA files using the trained model. 
Just replace the filenames for siRNA, shRNA, DNA, and new sequences with your own files to run the code with your own data.
