import nltk
import numpy as np
nltk .download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    '''
    split sentence into array of words/tokens
    A token can be a word or punctuation character, or number.
    '''
    return nltk.word_tokenize(sentence)


def stem(word):
    '''
    find root form of root word. 
    ex:
    words = ["organize", "organization", "organizing"]
    words = [stem(w) for w in words] 
    --> ["organ", "organ", "organ"]
    '''
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    #pass       # at the place of the pass i write this 
    """
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]     #ama je lakhyu hoi eej name lakhva nu badhe
        bog = [0, 1, 0, 1, 0, 0, 0]
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag



# jo aa akhis su to print thase aa            bog = [0, 1, 0, 1, 0, 0, 0]:::-->>
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]            # ahiya je lakhyu eeej badhue lakhva nu 
# bog = bag_of_words(sentence, words)
# print(bog)



# word = ["Organize", "Organization","Organizing", "helloqwerty", "how", "are", "you", "play"]
# strmmed_word = [stem(w) for w in word]
# print("Stemmed words:")
# print(strmmed_word)
























# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()


# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     return stemmer.stem(word.lower())

# def bag_of_words(tokenized_sentence, all_words):
#     pass


# a = "how are you"
# print(a)
# a = tokenize(a)
# print(a)