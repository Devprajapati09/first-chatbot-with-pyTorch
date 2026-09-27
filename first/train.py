import json
from nltk_utils import tokenize, stem, bag_of_words     #jo aa nanlakhu ne to error ave line number 15 ma 
import numpy as np


import torch
print("PyTorch version:", torch.__version__)
print("PyTorch loaded successfully!")

import torch.nn as nn
# form torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem, bag_of_words

from model import NeuralNet




with open('intension.json', 'r') as f:
    intents = json.load(f)

# print(intents) gere at the  place we use the undeer i have write 
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)       #CrossEntropyLoss   

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

#dataindex[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.n_samples


#hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
print(input_size, len(all_words))
print(output_size, tags)



dataset = ChatDataset() 
train_loader = DataLoader(dataset=dataset, batch_size=batch_size , shuffle=True, num_workers=2)


# model = NeuralNet(input_size=len(X_train[0]), hidden_size=8, num_classes=len(tags))
model = NeuralNet(input_size, hidden_size, output_size)  # Corrected line)




