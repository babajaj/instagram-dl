import json
import numpy as np

def getJSON(caption_file, images_file):
    with open(caption_file) as caps:
        captions = json.load(caps)
    with open(images_file) as imgs:
        images = np.array(json.load(imgs))
    return captions, images


data = getJSON("data/captions.json", "data/images.json")

print(data[1].shape)