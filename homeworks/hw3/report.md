# Report of CSCI-544 Homework 3: Part-of-Speech Tagging

## Python Printout

```
% python hw3.py
-------- Vocabulary Summary --------
Selected threshold for unknown words replacement: 2
Total size of the vocabulary: 23182
Total occurrence of <unk>: 20011

-------- HMM Model Summary --------
Number of transition parameters: 1351
Number of emission parameters: 30303
Model trained and saved to ./output/hmm.json.

-------- Greedy Decoding --------
Greedy decoding on the development data...
Total 5527, finished 0...200...400...600...800...1000...1200...1400...1600...1800...2000...2200...2400...2600...2800...3000...3200...3400...3600...3800...4000...4200...4400...4600...4800...5000...5200...5400...Done!

Writing predictions to ./output/greedy.out... Done!
Run 'python eval.py -p ./output/greedy.out -g ./data/dev' to evaluate the predictions!

-------- Viterbi Decoding --------
Viterbi decoding on the development data...
Total sentences: 5527
0...200...400...600...800...1000...1200...1400...1600...1800...2000...2200...2400...2600...2800...3000...3200...3400...3600...3800...4000...4200...4400...4600...4800...5000...5200...5400...Done!

Writing predictions to ./output/viterbi.out... Done!
Run 'python eval.py -p ./output/viterbi.out -g ./data/dev' to evaluate the predictions!

Evaluate the predictions? (y/n): y
total: 131768, correct: 122702, accuracy: 93.12%
total: 131768, correct: 124555, accuracy: 94.53%
```

## Task 1: Vocabulary Creation 
- Selected threshold for unknown words replacement: 2
- Total size of the vocabulary: 23182
- Total occurrence of <unk>: 20011
- Vocabulary saved to `./output/vocab.json`

## Task 2: HMM Model Training
- Number of transition parameters: 1351
- Number of emission parameters: 30303
- Model saved to `./output/hmm.json`

## Task 3: Greedy Decoding
- Accuracy on development data: 93.12%
- Output file: `./output/greedy.out`
  
## Task 4: Viterbi Decoding
- Accuracy on development data: 94.53%
- Output file: `./output/viterbi.out`
