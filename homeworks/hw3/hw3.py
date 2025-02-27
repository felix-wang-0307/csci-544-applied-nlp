import json

# Global configuration
train_file = "./data/train"  # Training file to READ
dev_file = "./data/dev"  # Development file to READ
test_file = "./data/test"  # Test file to READ
vocab_file = "./output/vocab.txt"  # Vocabulary file to WRITE
predicted_file = "./output/greedy.out"  # Prediction file to WRITE


def read_data(data_path) -> list[list[dict]]:
    """
    Args:
        data_path (str): The path to the training data file.
    Returns:
        list[list[dict]]: A list of sentences, where each sentence is a list of dictionaries.
            Each dictionary contains the following keys:
            - "index": The index of the word in the sentence.
            - "word": The word itself.
            - "tag": The part-of-speech tag of the word.
    """
    sentences = []
    current_sentence = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # Blank line indicates end of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split("\t")
                if len(parts) == 3:
                    index, word, tag = parts
                    current_sentence.append({
                        "index": index,
                        "word": word,
                        "tag": tag
                    })
        
        if current_sentence:
            sentences.append(current_sentence)  # Append last sentence if no blank line at the end

    return sentences


def create_vocabulary(sentences, vocab_file="vocab.txt"):
    # 1. Configuration
    THRESHOLD = 2  # Words less frequent than this will be replaced by <unk>

    # 2. Count word occurrences
    word_counts = {}
    for sentence in sentences:
        for entry in sentence:
            word = entry["word"]
            word_counts[word] = word_counts.get(word, 0) + 1

    # 3. Create vocabulary
    unk_count = 0
    words_to_remove = set()
    for word, count in word_counts.items():
        if count < THRESHOLD:
            # Replace word with <unk>
            unk_count += count
            words_to_remove.add(word)
    for word in words_to_remove:
        word_counts.pop(word)

    # 4. Sort words by descending count, then by alphabetical order
    words_list = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

    # 5. Write vocabulary to file
    with open(vocab_file, "w", encoding="utf-8") as f:
        f.write(f"<unk>\t{unk_count}\n")
        for word, count in words_list:
            f.write(f"{word}\t{count}\n")

    # 6. Print answers to questions
    print("-------- Vocabulary Summary --------")
    print(f"Selected threshold for unknown words replacement: {THRESHOLD}")
    print(f"Total size of the vocabulary: {len(words_list)}")
    print(f"Total occurrence of <unk>: {unk_count}")
    print()

    return word_counts, unk_count, words_list


def learn_hmm(sentences, word_counts, unk_count, output_file="./output/hmm.json"):
    """
    Learns HMM parameters (transition and emission probabilities) from the training data
    using the word counts and <unk> count from the vocabulary, and saves them in JSON format (hmm.json).
    """
    # Initialize data structures to store counts
    transitions_count = {}
    emission_count = {}
    tag_count = {}

    # Initialize counts for transitions and emissions
    for sentence in sentences:
        for i, entry in enumerate(sentence):
            word = entry["word"]
            tag = entry["tag"]

            # Handle <unk> replacement in words
            if word not in word_counts:
                word = "<unk>"

            # Count emissions (word -> tag)
            if tag not in emission_count:
                emission_count[tag] = {}
            emission_count[tag][word] = emission_count[tag].get(word, 0) + 1

            # Count tags
            tag_count[tag] = tag_count.get(tag, 0) + 1

            # Count transitions (tag -> next_tag)
            if i < len(sentence) - 1:
                next_tag = sentence[i + 1]["tag"]
                if tag not in transitions_count:
                    transitions_count[tag] = {}
                transitions_count[tag][next_tag] = (
                    transitions_count[tag].get(next_tag, 0) + 1
                )

    # Now compute transition and emission probabilities
    transition_probs = {}
    for tag in transitions_count:
        total_out_of_tag = sum(transitions_count[tag].values())
        for next_tag, count in transitions_count[tag].items():
            transition_probs[f"{tag}||{next_tag}"] = count / total_out_of_tag

    emission_probs = {}
    for tag in emission_count:
        total_tag = tag_count[tag]
        for word, count in emission_count[tag].items():
            emission_probs[f"{tag}||{word}"] = count / total_tag

    # Save the model to JSON
    hmm_model = {"transition": transition_probs, "emission": emission_probs}

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(hmm_model, json_file, indent=2)

    # Print a summary of transition and emission parameters
    print("-------- HMM Model Summary --------")
    print(f"Number of transition parameters: {len(transition_probs)}")
    print(f"Number of emission parameters: {len(emission_probs)}")
    print(f"Model trained and saved to {output_file}.")
    print()
    
    return transition_probs, emission_probs


def greedy_decode(sentences, transition_probs, emission_probs, unk_token="<unk>"):
    """
    Implements the greedy decoding algorithm to predict part-of-speech tags for sentences.
    """
    predictions = []
    
    for sentence in sentences:
        prev_tag = None
        sentence_pred = []
        
        for entry in sentence:
            word = entry["word"]
            # Handle unknown words (those not in the vocabulary)
            if word not in emission_probs:
                word = unk_token
            
            max_prob = 0
            best_tag = None
            
            # For the first word, use any tag with max emission probability
            if prev_tag is None:
                for tag in emission_probs:
                    prob = emission_probs.get(f"{tag}||{word}", 0)
                    if prob > max_prob:
                        max_prob = prob
                        best_tag = tag
            else:
                # For subsequent words, consider both transition and emission probabilities
                for tag in emission_probs:
                    transition_prob = transition_probs.get(f"{prev_tag}||{tag}", 0)
                    emission_prob = emission_probs.get(f"{tag}||{word}", 0)
                    prob = transition_prob * emission_prob
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_tag = tag
            
            sentence_pred.append((entry["index"], word, best_tag))
            prev_tag = best_tag
        
        predictions.append(sentence_pred)
    
    return predictions


def write_predictions_to_file(predictions, output_file):
    """
    Write the predicted POS tags to a file in the required format.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in predictions:
            for index, word, predicted_tag in sentence:
                f.write(f"{index}\t{word}\t{predicted_tag}\n")
            f.write("\n")  # Blank line to separate sentences


def evaluate_predictions(predicted_file, gold_standard_file):
    """
    Evaluates the predictions using the eval.py script.
    """
    import os
    os.system(f"python eval.py -p {predicted_file} -g {gold_standard_file}")


if __name__ == "__main__":
    # Read the training data once
    sentences = read_data(train_file)
    
    # Create vocabulary using the training data
    word_counts, unk_count, words_list = create_vocabulary(sentences, vocab_file)
    
    # Train HMM using the same data
    transition_probs, emission_probs = learn_hmm(sentences, word_counts, unk_count)
    
    # Greedy decoding on the development data
    dev_sentences = read_data(dev_file)
    dev_predictions = greedy_decode(dev_sentences, transition_probs, emission_probs)
    write_predictions_to_file(dev_predictions, predicted_file)
