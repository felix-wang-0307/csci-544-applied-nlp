import os
import json

# Global configuration
train_file = "./data/train"  # Training file to READ
dev_file = "./data/dev"  # Development file to READ
test_file = "./data/test"  # Test file to READ
if not os.path.exists("./output"):
    os.makedirs("./output")
vocab_file = "./output/vocab.txt"  # Vocabulary file to WRITE
greedy_predicted_file = (
    "./output/greedy.out"  # Predicted POS tags using greedy decoding to WRITE
)
viterbi_predicted_file = (
    "./output/viterbi.out"  # Predicted POS tags using Viterbi decoding to WRITE
)


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
                    current_sentence.append({"index": index, "word": word, "tag": tag})

        if current_sentence:
            sentences.append(
                current_sentence
            )  # Append last sentence if no blank line at the end

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
    # Extract known words from emission_probs keys (they are in the format "tag||word")
    known_words = set()
    for key in emission_probs.keys():
        word = key.split("||")[1]
        known_words.add(word)

    predictions = []
    valid_tags = set()
    for key in emission_probs.keys():
        tag = key.split("||")[0]
        valid_tags.add(tag)

    print("-------- Greedy Decoding --------")
    print("Greedy decoding on the development data...", flush=True)
    print(f"Total {len(sentences)}, finished ", end="", flush=True)

    for i, sentence in enumerate(sentences):
        prev_tag = None
        sentence_pred = []

        for entry in sentence:
            word = entry["word"]
            # Correct check for unknown words
            if word not in known_words:
                word = unk_token

            max_prob = 0
            best_tag = None

            if prev_tag is None:
                # For the first word, select the tag with the highest emission probability
                for tag in valid_tags:
                    prob = emission_probs.get(f"{tag}||{word}", 0)
                    if prob > max_prob:
                        max_prob = prob
                        best_tag = tag
            else:
                # For subsequent words, use both transition and emission probabilities
                for tag in valid_tags:
                    transition_prob = transition_probs.get(
                        f"{prev_tag}||{tag}", 1e-10
                    )  # Avoid zero probabilities
                    emission_prob = emission_probs.get(f"{tag}||{word}", 1e-10)
                    prob = transition_prob * emission_prob
                    if prob > max_prob:
                        max_prob = prob
                        best_tag = tag

            # Fallback in case no tag is found
            if best_tag is None:
                print(
                    f"WARNING: No best tag found for '{word}', assigning 'NN' as default."
                )
                best_tag = "NN"

            sentence_pred.append((entry["index"], word, best_tag))
            prev_tag = best_tag

        predictions.append(sentence_pred)
        if i % 200 == 0:
            print(f"{i}...", end="", flush=True)

    print("Done!\n")

    return predictions


def write_predictions_to_file(predictions, output_file):
    """
    Write the predicted POS tags to a file in the required format.
    """
    print("Writing predictions to", output_file, end="... ", flush=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in predictions:
            for index, word, predicted_tag in sentence:
                f.write(f"{index}\t{word}\t{predicted_tag}\n")
            f.write("\n")  # Blank line to separate sentences
    print("Done!")
    
    print(f"Run 'python eval.py -p {output_file} -g {dev_file}' to evaluate the predictions!\n")


def viterbi_decode(sentences, transition_probs, emission_probs, unk_token="<unk>"):
    """
    Implements the Viterbi decoding algorithm to predict part-of-speech tags for sentences.
    """
    # Extract known words and valid tags from emission_probs keys
    known_words = set()
    for key in emission_probs.keys():
        word = key.split("||")[1]
        known_words.add(word)

    valid_tags = set()
    for key in emission_probs.keys():
        tag = key.split("||")[0]
        valid_tags.add(tag)

    predictions = []
    print("-------- Viterbi Decoding --------")
    print("Viterbi decoding on the development data...", flush=True)
    print(f"Total sentences: {len(sentences)}", flush=True)

    for i, sentence in enumerate(sentences):
        n = len(sentence)
        V = (
            {}
        )  # V[t][tag] stores the highest probability for a tag sequence ending in tag at position t.
        backpointer = (
            {}
        )  # backpointer[t][tag] stores the best previous tag for tag at position t.

        # Initialization for t = 0 (first word)
        V[0] = {}
        backpointer[0] = {}
        word = sentence[0]["word"]
        if word not in known_words:
            word = unk_token
        for tag in valid_tags:
            # No start probability is given, so we rely solely on the emission probability.
            emission_prob = emission_probs.get(f"{tag}||{word}", 1e-10)
            V[0][tag] = emission_prob
            backpointer[0][tag] = None

        # Recursion: fill in V[t] for t = 1, 2, ..., n-1
        for t in range(1, n):
            V[t] = {}
            backpointer[t] = {}
            word = sentence[t]["word"]
            if word not in known_words:
                word = unk_token
            for tag in valid_tags:
                max_prob = 0
                best_prev_tag = None
                # Consider all possible previous tags
                for prev_tag in valid_tags:
                    trans_prob = transition_probs.get(f"{prev_tag}||{tag}", 1e-10)
                    prob = V[t - 1][prev_tag] * trans_prob
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag
                emission_prob = emission_probs.get(f"{tag}||{word}", 1e-10)
                V[t][tag] = max_prob * emission_prob
                backpointer[t][tag] = best_prev_tag

        # Termination: pick the tag with the highest probability at the final position
        max_final_prob = 0
        best_last_tag = None
        for tag in valid_tags:
            if V[n - 1][tag] > max_final_prob:
                max_final_prob = V[n - 1][tag]
                best_last_tag = tag

        # Backtrace to recover the best tag sequence
        best_tags = [None] * n
        best_tags[n - 1] = best_last_tag
        for t in range(n - 1, 0, -1):
            best_tags[t - 1] = backpointer[t][best_tags[t]]

        # Build the prediction for the current sentence
        sentence_pred = []
        for t, entry in enumerate(sentence):
            # Again, ensure unknown words are replaced
            word = entry["word"]
            if word not in known_words:
                word = unk_token
            sentence_pred.append((entry["index"], word, best_tags[t]))
        predictions.append(sentence_pred)

        if i % 200 == 0:
            print(f"{i}...", end="", flush=True)

    print("Done!")

    return predictions


if __name__ == "__main__":
    # Read the training data once
    train_sentences = read_data(train_file)

    # Create vocabulary using the training data
    word_counts, unk_count, words_list = create_vocabulary(train_sentences, vocab_file)

    # Train HMM using the same data
    transition_probs, emission_probs = learn_hmm(
        train_sentences, word_counts, unk_count
    )

    # Greedy decoding on the development data
    dev_sentences = read_data(dev_file)
    dev_predictions = greedy_decode(dev_sentences, transition_probs, emission_probs)
    write_predictions_to_file(dev_predictions, greedy_predicted_file)

    # Viterbi decoding on the development data
    viterbi_predictions = viterbi_decode(
        dev_sentences, transition_probs, emission_probs
    )
    write_predictions_to_file(viterbi_predictions, viterbi_predicted_file)

    evaluate = input("Evaluate the predictions? (y/n): ")
    if evaluate.lower() == "y":
        os.system(f"python eval.py -p {greedy_predicted_file} -g {dev_file}")
        os.system(f"python eval.py -p {viterbi_predicted_file} -g {dev_file}")
