# This file implements a pipeline for training a BiLSTM model for Named Entity Recognition (NER) 
# GloVe embeddings are used for the model. 
import os

if __name__ == "__main__":
    # Train the model
    print("--------------- Training the model with GloVe --------------------")
    os.system("python src/train.py --use_glove -t data/train -d data/dev --save_model_path out/blstm2.pt --epochs 50 --batch_size 32 --learning_rate 0.1")
    # Predict on dev data
    print("\n--------------- Predicting on dev data ---------------------------")
    os.system("python src/predict.py --use_glove --has_tag -i data/dev -m out/blstm2.pt -o out/dev2.out")
    # Evaluate the predictions on dev data
    print("\n--------------- Evaluating the predictions ------------------------")
    os.system("python eval.py -p out/dev2.out -g data/dev")
    # Predict on test data
    print("\n--------------- Predicting on test data ---------------------------")
    os.system("python src/predict.py --use_glove -i data/test -m out/blstm2.pt -o out/test2.out")