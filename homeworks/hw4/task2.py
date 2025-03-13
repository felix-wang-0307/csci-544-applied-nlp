# This file implements a pipeline for training a BiLSTM model for Named Entity Recognition (NER) 
# GloVe embeddings are used for the model. 
import os

if __name__ == "__main__":
    # Train the model
    os.system("python src/train.py --use_glove -t data/train -d data/dev --save_model_path out/blstm2.pt --epochs 10 --batch_size 32 --learning_rate 0.1")