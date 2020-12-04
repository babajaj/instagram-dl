import tensorflow as tf
import numpy as np
import nltk
from preprocess import tokenize, pre_image
from unscraper import load_data
from tensorflow.keras.layers import (LSTM, Embedding, Input, Dense, Dropout, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from nltk.translate.bleu_score import sentence_bleu



class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next character in a sequence.

        :param vocab_size: The number of unique characters in the data
        """
        self.caption_length = 150
        self.batch_size = 100
        self.vocab_size = vocab_size
        self.embedding_size = 256
        self.learning_rate = 0.01
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        ##caption model
        self.embedding = Embedding(self.vocab_size, self.embedding_size, mask_zero=True)
        self.encoder = LSTM(self.embedding_size)
        self.dropout_caps = Dropout(0.5)
        
        ##images model
        self.dropout_imgs = Dropout(0.5)
        self.dense_imgs = Dense(self.embedding_size)

        ##merge
        self.add = Add()
        self.dense = Dense(self.embedding_size, activation='relu')
        self.predict = Dense(self.vocab_size, activation='softmax')
        



    def call(self, images, captions, initial_state):
        """
        :param inputs: character ids of shape (batch_size, window_size)
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        ##caps
        embeddings = self.embedding(captions)
        encode = self.dropout_caps(encode)
        whole_seq_output, final_memory_state, final_carry_state  = self.encoder(encode)

        ##images
        images = self.dropout_imgs(images)
        images = self.dense_imgs(images)

        ##merge
        combined = self.add(whole_seq_output, images)
        combined = self.dense(combined)
        output = self.predict(combined)
        
        return output, (final_memory_state,final_carry_state)

    def loss(self, probs, labels, mask):
        """

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels,probs[0])
        loss = tf.reduce_sum(loss * mask)
        return loss

def train(model, images, captions):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
   
    remainderInputs = len(images) % model.caption_length
    currInputs = images[:-remainderInputs]
    
    remainderLabels = len(captions) % model.caption_length
    currLabels = captions[:-remainderLabels]
    
    currInputs = tf.reshape(currInputs,(-1,model.caption_length))
    currLabels = tf.reshape(currLabels,(-1,model.caption_length))

    for i in range(0, len(currInputs)//model.batch_size): 
        inputs = currInputs[i*model.batch_size:(i+1)* model.batch_size]
        labels = currLabels[i*model.batch_size:(i+1)* model.batch_size]
        with tf.GradientTape() as tape:
            predictions = model.call(inputs, labels, None) 
            loss = model.loss(predictions, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return None
    


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    


def generate_caption(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    IDK IF THIS ONE IS NEEDED
    it is because we need to generate captions to see our results
    """

    reverse_vocab = {idx: char for char, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    data = load_data("captions.json","images.npy")
    training_captions, vocab_dict = tokenize(data[0])
    images = data[1]
    model = Model(len(vocab_dict))
    train(model, images, training_captions)



if __name__ == '__main__':
    main()

   