import random 
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NeuralNet(input_size=len(X_train[0]), hidden_size=8, num_classes=len(tags))

with open('intension.json', 'r') as f:
        intents = json.load(f)

FILE= "data.pth"
data = torch.load(FILE)


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Corrected line)
model.load_state_dict(model_state)
model.eval()


botname = "Hacker"

def get_response(msg):
    sentence = tokenize(msg)           #here it defines from the train.py file, it is a function that ''' tokenizes ''' the sentence into words
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])           #reshapes the array into a 2D array with one row and as many columns as there are elements in the original array
    X = torch.from_numpy(X) #X = torch.from_numpy(X).to(device)


    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]            #define from the intension.json file, it is a list of tags that correspond to the intents 


    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


   
    if prob.item() > 0.75:                    #if the probability of the predicted tag is greater than 0.75, it will print a random response from the list of responses for that tag     
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])                #define from the intension.json file,  it is a list of tags that correspond to the intents 

    return "I do not understand..."

    
