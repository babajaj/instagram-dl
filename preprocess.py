import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from unscraper import load_data
import numpy as np

def tokenize(captions):
    """
    :param captions: Captions of all the images
    :return: Tuple of train (1-d list or array with each character from captions in id form), vocabulary (Dict containg word->index mapping)
    """
    vocab_dict = dict()
    x = 1
    #stop word is 0
    training_data = []
    counter = 0
    string = ""
    for line in captions:
        line = line.lower()
        padding = 0
        training_data.append(-1) #start token
        for c in line:
            if (c not in vocab_dict):
                vocab_dict[c] = x
                x = x + 1
            padding += 1
            training_data.append(vocab_dict[c])
            if padding == 150:
                break
        training_data.append(-2) #end token    
        while padding < 150:
            training_data.append(0)
            padding += 1
    return training_data, vocab_dict
    

# data = load_data("data/captions.json","data/images.npy")
# captions = tokenize(data[0])
# print(captions[1])
cap = ["folklore will have 16 songs on the standard edition, but the physical deluxe editions will include a bonus track called \u201cthe lakes.\u201d Because this is my 8th studio album, I made 8 deluxe CD editions and 8 deluxe vinyl editions that are available for one week\ud83d\ude04 Each deluxe edition has unique covers, photos, and artwork. Available exclusively at taylorswift.com ","goodbye!"]
out = tokenize(cap)
print(out[0])
print(len(out[0]))

def pre_image(images):
    features = []
    # Model to pre-process images
    cnn_model = VGG16()
    # re-structure the model
    cnn_model.layers.pop()
    cnn_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
    x = preprocess_input(images)
    features = cnn_model.predict(x)
    with open("data/features.npy", "wb") as feats:
        np.save(feats, features, allow_pickle=True)
    return features



