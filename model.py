import tensorflow as tf
import numpy as np
import nltk
from preprocess import tokenize, pre_image
from unscraper import load_data
from tensorflow.keras.layers import (LSTM, Embedding, Input, Dense, Dropout, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt



class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        """
        The Model class predicts the next character in a sequence.

        :param vocab_size: The number of unique characters in the data
        """
        self.caption_length = 152
        self.batch_size = 10
        self.vocab_size = vocab_size
        self.embedding_size = 30
        self.encoder_size = 128
        self.learning_rate = 0.01
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        ##caption model
        self.embedding = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1))
        self.encoder = LSTM(self.encoder_size, return_sequences=True, return_state=True)
        self.dropout_caps = Dropout(0.5)
        
        ##images models
        self.dropout_imgs = Dropout(0.5)
        self.dense_imgs = Dense(self.encoder_size)

        ##merge
        self.dense = Dense(self.encoder_size, activation='relu')
        self.predict = Dense(self.vocab_size, activation='softmax')
        self.printed = False
        



    def call(self, images, captions, initial_state):
        """
        :param inputs: character ids of shape (batch_size, window_size)
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        ##caps
        captions = tf.convert_to_tensor(captions)
        embeds = tf.nn.embedding_lookup(self.embedding, captions)
        encode = self.dropout_caps(embeds)
        whole_seq_output, final_memory_state, final_carry_state  = self.encoder(encode, initial_state=initial_state)
        
        ##images
        images = self.dropout_imgs(images)
        images = self.dense_imgs(images)
      
        ##merge
        combined = whole_seq_output + tf.expand_dims(images, 1)
        combined = self.dense(combined)
        print(tf.reduce_sum(combined))
        output = self.predict(combined)
        
        return output, (final_memory_state,final_carry_state)

    def loss(self, probs, labels, mask):
        """

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        loss = tf.reduce_sum(loss * mask)
        print(f'Loss: {loss}')
        return loss

def train(model, images, captions):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
   
    images = images[:-(len(images) % model.batch_size)]
    captions = np.reshape(captions, (-1, model.caption_length))
    captions = captions[:-(len(captions) % model.batch_size)]
    caption_input = captions[:, :-1]
    caption_label = captions[:, 1:]
    loss_graph = []

    
    for i in range(0, len(images)//model.batch_size-1): 
        print(i)
        imgs = images[i*model.batch_size:(i+1)* model.batch_size]
        caps_input = caption_input[i*model.batch_size:(i+1)* model.batch_size]
        caps_label = caption_label[i*model.batch_size:(i+1)* model.batch_size]
        with tf.GradientTape() as tape:
            predictions = model.call(imgs, caps_input, None)
            padding_mask = np.where(caps_label == 0, 0, 1)
            loss = model.loss(predictions[0], caps_label, padding_mask)
            loss_graph.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_graph
    


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
def accuracy_function(self, prbs, labels, mask):
    """
    Computes the batch accuracy
    
    :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :param mask:  tensor that acts as a padding mask [batch_size x window_size]
    :return: scalar tensor of accuracy of the batch between 0 and 1
    """

    decoded_symbols = tf.argmax(input=prbs, axis=2)
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
    return accuracy

def generate_caption(vocab, image, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    it is because we need to generate captions to see our results
    """



    reverse_vocab = {idx: char for char, idx in vocab.items()}
    previous_state = None

    first_char = 1
    next_input = [[first_char]]
    text = ""
    out_index = 0
    i = 0
    while out_index != 2 and i < 200:
        i +=1 
        logits, previous_state = model.call(image, next_input, previous_state)
        # logits = np.array(logits[0,0,:])
        # top_n = np.argsort(logits)[-sample_n:]
        # n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.argmax(logits)
        if out_index != 2:
            print("    " + str((reverse_vocab[out_index])))
            text += (reverse_vocab[out_index])
        next_input = [[out_index]]
    print(text)


def vizualize_loss(loss_arr):
     plt.plot(np.arange(len(loss_arr)), loss_arr)
     plt.ylabel('loss')
     plt.xlabel('batch #')
     plt.show()


def main():
    data = load_data("data/captions.json","data/features.npy")
    training_captions, vocab_dict = tokenize(data[0])
    images = data[1]
    print(np.max(training_captions))
    model = Model(len(vocab_dict))
    for i in range(50):
        loss_graph = train(model, images, training_captions)
        generate_caption(vocab_dict, np.array([images[0]]), model)
        vizualize_loss(loss_graph)




if __name__ == '__main__':
    main()

   