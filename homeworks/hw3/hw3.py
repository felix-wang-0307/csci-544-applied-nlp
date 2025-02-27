import json

# Global configuration
train_file = "./data/train"  # Training file to READ
# vocab_file = "vocab.txt"  # Vocabulary file to WRITE

def read_data(data_path):
    """
    Reads the training file and stores it in memory.
    Returns a list of sentences where each sentence is a list of (word, tag) tuples.
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
                    _, word, tag = parts
                    current_sentence.append((word, tag))
        
        if current_sentence:
            sentences.append(current_sentence)  # Append last sentence if no blank line at the end

    return sentences


def create_vocabulary(sentences, vocab_file="vocab.txt"):
    """
    Creates a vocabulary from the training sentences and writes it to a file.
    """
    # 1. Configuration
    THRESHOLD = 2  # Words less frequent than this will be replaced by <unk>

    # 2. Count word occurrences
    word_counts = {}
    for sentence in sentences:
        for word, tag in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1

    # 3. Create vocabulary (replace words below threshold with <unk>)
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
    print(f"Selected threshold for unknown words replacement: {THRESHOLD}")
    print(f"Total size of the vocabulary: {len(words_list)}")
    print(f"Total occurrence of <unk>: {unk_count}")

    return {
        "word_counts": word_counts,
        "unk_count": unk_count,
        "words_list": words_list,
    }


def learn_hmm(sentences, word_counts, unk_count, output_file="hmm.json"):
    """
    Learns HMM parameters (transition and emission probabilities) from the training sentences
    using the word counts and <unk> count from the vocabulary, and saves them in JSON format.
    """
    # Initialize data structures to store counts
    transitions_count = {}
    emission_count = {}
    tag_count = {}

    # Initialize counts for transitions and emissions
    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence):
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
                _, next_tag = sentence[i + 1]
                if tag not in transitions_count:
                    transitions_count[tag] = {}
                transitions_count[tag][next_tag] = transitions_count[tag].get(next_tag, 0) + 1

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
    print(f"Number of transition parameters: {len(transition_probs)}")
    print(f"Number of emission parameters: {len(emission_probs)}")
    print(f"Model trained and saved to {output_file}.")


if __name__ == "__main__":
    # Read the data once
    sentences = read_data(train_file)

    # Create vocabulary
    vocab_data = create_vocabulary(sentences)

    # Train HMM model
    learn_hmm(sentences, vocab_data["word_counts"], vocab_data["unk_count"])
