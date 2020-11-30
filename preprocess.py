

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
        for c in line:
            if c == "\\":
                string = "\\"
                counter += 1
            elif counter > 0:
                string += c
                counter += 1
            elif counter == 6 and (string not in vocab_dict) :
                counter = 0
                vocab_dict[string] = x
                string = ""
                x = x + 1
            elif c not in vocab_dict:
                vocab_dict[c] = x
                x = x + 1
            training_data.append(vocab_dict[c])
        training_data.append(0)
    return training_data, vocab_dict
    print(len(training_data))


captions = [
    "rum punch + the beach ", "hello fall \ud83d\udc9a "]
print(tokenize(captions))