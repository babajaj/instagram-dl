from preprocess import tokenize, pre_image
from unscraper import load_data
from tensorflow.keras.layers import (LSTM, Embedding, Input, Dense, Dropout, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Model(tf.keras.Model):
    def __init__(self, vocab_size, caption_length):
        """
        The Model class predicts the next character in a sequence.

        :param vocab_size: The number of unique characters in the data
        """
        
        self.batch_size = 100
        self.vocab_size = vocab_size
        self.caption_length = caption_length
        
        ##caption model
        input_caps = Input([self.batch_size, self.caption_length])
        dropout_caps = Dropout(0.5)
        ############ TODO: LSTM #################

        ##images model
        input_imgs = Input([self.batch_size, 4096])
        dropout_imgs = Dropout(0.5)
        dense_imgs = Dense(256)

        ##merge
        add = Add()
        dense = Dense(256, activation='relu')
        predict = Dense(self.vocab_size)
        



    def call(self, inputs, initial_state):
        """
        :param inputs: character ids of shape (batch_size, window_size)
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """

    def loss(self, probs, labels):
        """

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
   
    print(train_labels.shape)
    remainderInputs = len(train_inputs) % model.window_size
    currInputs = train_inputs[:-remainderInputs]
    
    remainderLabels = len(train_labels) % model.window_size
    currLabels = train_labels[:-remainderLabels]
    
    currInputs = tf.reshape(currInputs,(-1,model.window_size))
    currLabels = tf.reshape(currLabels,(-1,model.window_size))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for i in range(0, len(currInputs)//model.batch_size): 
        inputs = currInputs[i*model.batch_size:(i+1)* model.batch_size]
        labels = currLabels[i*model.batch_size:(i+1)* model.batch_size]
        with tf.GradientTape() as tape:
            predictions = model.call(inputs,None) 
            loss = model.loss(predictions, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return None
    


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    IDK IF THIS ONE IS NEEDED
    """

 


def main():
    data = load_data("captions.json","images.npy")
    captions = tokenize(data[0])
    images = pre_image(data[1])

if __name__ == '__main__':
    main()

   