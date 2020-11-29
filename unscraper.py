import json
import numpy as np

def load_data(caption_file, images_file):
    with open(caption_file) as caps:
        captions = json.load(caps)
    print("loaded captions")
    with open(images_file, 'rb') as imgs:
        images = np.load(imgs, allow_pickle=True)
    print('loaded images')
    return captions, images

data = load_data("data/captions.json", "data/images.npy")

print(data[1])