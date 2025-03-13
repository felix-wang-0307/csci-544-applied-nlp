# Report of CSCI-544 Homework 4 - NER (Named Entity Recognition)

## Python Printout
### Task 1
```
% python task1.py
-------------------- Training the model --------------------
Loading datasets data/train and data/dev...
Training model...
Epoch 1/25 Loss: 0.6111
Epoch 2/25 Loss: 0.3517
Epoch 3/25 Loss: 0.2244
...
Epoch 21/25 Loss: 0.0005
Epoch 22/25 Loss: 0.0004
Epoch 23/25 Loss: 0.0004
Epoch 24/25 Loss: 0.0004
Epoch 25/25 Loss: 0.0004
Model saved to out/blstm1.pt

----------------- Predicting on dev data --------------------
Loading dataset from data/dev...
Predicting...
Predictions saved to out/dev1.out

----------------- Evaluating the predictions -----------------
processed 51578 tokens with 5942 phrases; found: 5522 phrases; correct: 4145.
accuracy:  94.59%; precision:  75.06%; recall:  69.76%; FB1:  72.31
              LOC: precision:  87.36%; recall:  78.23%; FB1:  82.54  1645
             MISC: precision:  79.09%; recall:  73.86%; FB1:  76.39  861
              ORG: precision:  65.23%; recall:  65.47%; FB1:  65.35  1346
              PER: precision:  68.80%; recall:  62.38%; FB1:  65.43  1670

----------------- Predicting on test data ---------------------
Loading dataset from data/test...
Predicting...
Predictions saved to out/test1.out
```
### Task 2 (50 epochs)
```
% python task2.py
--------------- Training the model with GloVe --------------------
Loading datasets data/train and data/dev...
Loading GloVe embeddings from {glove_path}...
Training model...
Epoch 1/50 Loss: 0.4340
Epoch 2/50 Loss: 0.2463
Epoch 3/50 Loss: 0.1818
Epoch 4/50 Loss: 0.1400
Epoch 5/50 Loss: 0.1115
Epoch 6/50 Loss: 0.0887
Epoch 7/50 Loss: 0.0708
...
Epoch 47/50 Loss: 0.0003
Epoch 48/50 Loss: 0.0003
Epoch 49/50 Loss: 0.0003
Epoch 50/50 Loss: 0.0003
Model saved to out/blstm2.pt

--------------- Predicting on dev data ---------------------------
Loading dataset from data/dev...
Loading GloVe embeddings from ./data/glove.6B.100d...
Predicting...
Predictions saved to out/dev2.out

--------------- Evaluating the predictions ------------------------
processed 51578 tokens with 5942 phrases; found: 6161 phrases; correct: 4869.
accuracy:  96.60%; precision:  79.03%; recall:  81.94%; FB1:  80.46
              LOC: precision:  87.40%; recall:  85.74%; FB1:  86.56  1802
             MISC: precision:  72.19%; recall:  79.39%; FB1:  75.62  1014
              ORG: precision:  69.00%; recall:  77.33%; FB1:  72.93  1503
              PER: precision:  82.79%; recall:  82.79%; FB1:  82.79  1842

--------------- Predicting on test data ---------------------------
Loading dataset from data/test...
Loading GloVe embeddings from ./data/glove.6B.100d...
Predicting...
Predictions saved to out/test2.out
```

## Result Evaluation
### Task 1
The model trained without GloVe embeddings achieved an F1 score of 0.72 and an accuracy of 94.59% on the dev data, after 25 epochs. 

The model seems to converge after 25 epochs, as the loss is 0.0004 in the last epoch.

### Task 2
The model trained with GloVe embeddings achieved an F1 score of 0.80 and an accuracy of 96.60% on the dev data.

The model seems to converge after 50 epochs, as the loss is 0.0003 in the last epoch.
## Analysis
Introduction of GloVe embeddings improved the model performance significantly. The F1 score increased from 0.72 to 0.80, and the accuracy increased from 94.59% to 96.60%.

A momentum is observed in the model trained with GloVe embeddings. This indicates that the model is learning more efficiently with the embeddings.
