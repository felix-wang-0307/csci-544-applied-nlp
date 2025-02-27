# CSCI-544 Homework 3: Part-of-Speech Tagging

## File Structure
The file structure is expected to be as follows:
```
- hw3
  - data
    - train
    - test
    - dev
  - hw3.py
  - output
    - ... (output files by the code)
  - README.md
```
Make sure to put `data` directory in the same directory as the code. As it is too large to be uploaded, I have not included it in the submission.

## Running & Evaluation
### Requirements
- Python 3.8+ (Because I have used `f-strings` and type hints)
- **No external libraries** are used in this code - Yes! I implemented everything using raw Python to avoid endless virtual envs!!

### Data Path
- We will use the Wall Street Journal section of the Penn Treebank to build an HMM model for part-of-speech tagging. 
- The code expects the data to be present in the `data` directory in the same directory as the code.
- The `data` directory should contain the following files:
  - `train`: The training data
  - `test`: The test data
  - `dev`: The development data


### Running the Code
- The code can be run using the following command:
  ```bash
  python hw3.py
  ```
- The code will generate the output files in the `output` directory.
  
### Evaluation
- The code will generate the output files in the `output` directory.
- The evaluation can be done using the following command:
  ```bash
  python eval.py −p {predicted file} −g {gold-standard file}
  ```


