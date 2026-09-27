import nltk
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


# Tokenization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Stemming
def stem(word):
    return stemmer.stem(word.lower())


# Bag of Words
def bag_of_words(tokenized_sentence, all_words):
    # Stem all words in the sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Create an array of zeros
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Put 1 where the word exists
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


# Test
a = "how are you"

print("Original sentence:")
print(a)

a = tokenize(a)

print("Tokens:")
print(a)

print("Stem test:")
print(stem("playing"))
print(stem("played"))
print(stem("plays"))


# Bag of Words test
all_words = ["hello", "how", "are", "you", "play"]

bag = bag_of_words(a, all_words)

print("Bag of Words:")
print(bag)