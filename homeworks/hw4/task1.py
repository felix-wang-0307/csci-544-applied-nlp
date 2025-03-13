# This file implements a pipeline for training a BiLSTM model for Named Entity Recognition (NER) 
# GloVe embeddings are NOT used for the model. 
import os

if __name__ == "__main__":
    # Train the model
    print("-------------------- Training the model --------------------")
    os.system("python src/train.py -t data/train -d data/dev --save_model_path out/blstm1.pt --epochs 25 --batch_size 32 --learning_rate 0.1")
    # Predict on dev data
    print("\n----------------- Predicting on dev data --------------------")
    os.system("python src/predict.py -i data/dev --has_tag -m out/blstm1.pt -o out/dev1.out")
    # Evaluate the predictions on dev data
    print("\n----------------- Evaluating the predictions -----------------")
    os.system("python eval.py -p out/dev1.out -g data/dev")
    # Predict on test data
    print("\n----------------- Predicting on test data ---------------------")
    os.system("python src/predict.py -i data/test -m out/blstm1.pt -o out/test1.out")