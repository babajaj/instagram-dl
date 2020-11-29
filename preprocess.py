from keras.layers import (LSTM, Embedding, Input, Dense, Dropout)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from unscrapper.py import load_data
def tokenize:
    file1 = open(train_file, 'r') 
    Lines = file1.readlines()
    training_data =[]
    x = 0
    vocab_dict = dict()
    for line in Lines: 
        arr = line.strip().split()
        for i in range(len(arr)):
            if(arr[i] not in vocab_dict):
                vocab_dict[arr[i]] = x
                x = x + 1
            training_data.append(vocab_dict[arr[i]])