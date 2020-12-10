import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from unscraper import load_data
import numpy as np
from PIL import Image

def tokenize(captions):
    """
    :param captions: Captions of all the images
    :return: Tuple of train (1-d list or array with each character from captions in id form), vocabulary (Dict containg word->index mapping)
    """
    vocab_dict = {'P': 0, 'S': 1, 'T': 2, 'X': 3}
    freq = {}
    x = 4
    training_data = []
    counter = 0
    string = ""
    
    for line in captions:
        line = line.lower()
        for c in line:
            if c not in freq:
                freq[c] = 0
            freq[c] += 1
    unk_c = []
    for c in freq:
        if freq[c] < 50:
            unk_c.append(c)
    for line in captions:
        line = line.lower()
        padding = 0
        training_data.append(1) #start token
        for c in line:
            if c not in unk_c and c not in vocab_dict:
                vocab_dict[c] = x
                x = x + 1
            padding += 1
            training_data.append(vocab_dict[c] if c in vocab_dict else 3)
            if padding == 150:
                break
        training_data.append(2) #end token    
        while padding < 150:
            training_data.append(0)
            padding += 1
    return training_data, vocab_dict
    

def pre_image(images):
    features = []
    # Model to pre-process images
    cnn_model = VGG16()
    # re-structure the model
    cnn_model.layers.pop()
    cnn_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
    x = preprocess_input(images)
    features = cnn_model.predict(x)
    with open("data/fun.npy", "wb") as feats:
        np.save(feats, features, allow_pickle=True)
    return features


image = Image.open('/data/fun_pic.png')
pre_image(image)


# data, dicti = tokenize(load_data("data/captions.json","data/features.npy")[0])
# print(len(data) / 152)
