import os
import re
import gensim
import logging

from glob import glob
from collections import Counter
from gensim import downloader as api

from keras.models import Sequential, load_model
from keras.layers import LSTM, Bidirectional, Dropout, Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

DATA_DIR = "/Users/miMacbookPro/Documents/GitMercurial/what-would-gordon-do/data/processed_data/"
MODEL_DIR = "/Users/miMacbookPro/Documents/GitMercurial/what-would-gordon-do/models/"
W2V_FILENAME = "w2v-google.model"

EMBEDDING_DIM = 128
SEQUENCE_LENGTH = 3
RNN_SIZE = 256 # size of RNN

LEARNING_RATE = 1.e-3
BATCH_SIZE = 512
NUM_EPOCHS = 100

TEST_SEED = ["what", "the", "fuck"]
TEST_NUM = 5

def main():
    file_list = glob(DATA_DIR+"*.txt")
    documents = list(read_input(file_list))
    logging.info("Done reading data file")

    w2v_model_path = os.path.join(MODEL_DIR, W2V_FILENAME)
    if os.path.isfile(w2v_model_path):
        logging.info("Loading model from {}".format(w2v_model_path))
        model = gensim.models.Word2Vec.load(w2v_model_path)
    else:
        logging.info("Training model")
        pretrain_corpus = api.load('wiki-english-20171001')
        dataset_size = api.info('wiki-english-20171001')["file_size"]
        model = gensim.models.Word2Vec(pretrain_corpus,
                                       size=EMBEDDING_DIM,
                                       window=SEQUENCE_LENGTH,
                                       min_count=2,
                                       workers=10)
        model.train(pretrain_corpus,
                    total_examples=dataset_size,
                    epochs=1)
        logging.info("Saving model to {}".format(w2v_model_path))
        model.save(w2v_model_path)
        model.train(documents,
                    total_examples=len(documents),
                    epochs=100)

        logging.info("Saving model to {}".format(w2v_model_path))
        model.save(w2v_model_path)

    w1 = "fuck"
    print("Most similar to {0}".format(w1),
          model.wv.most_similar(positive=w1)
          )
    w1 = "shit"
    print("Most similar to {0}".format(w1),
          model.wv.most_similar(positive=w1,
                                topn=6)
          )

    print("Vector representation of {0} is {1}".format(w1, model.wv[w1]))

    gordon_text = get_gordon(file_list)
    sequences = []
    next_word = []
    word_count = 0
    for line in gordon_text:
        if len(line) > SEQUENCE_LENGTH:
            try:
                line_vec = model.wv[line]
                for s, seq_end in enumerate(range(SEQUENCE_LENGTH, len(line_vec))):
                    sequences.append(line_vec[s:seq_end])
                    next_word.append(line_vec[seq_end])
            except KeyError:
                logging.warning("Skipping line as it contains a word not found in the dictionary")
    seq_array = np.array(sequences)
    next_word_array = np.array(next_word)
    print(seq_array.shape)
    print(next_word_array.shape)

    lstm_model_path = os.path.join(MODEL_DIR, "lstm.h5")
    if os.path.isfile(lstm_model_path):
        logging.info("Loading model from {}".format(lstm_model_path))
        lstm_model = load_model(lstm_model_path)
    else:
        logging.info("Traing model...")
        lstm_model = bidirectional_lstm_model(SEQUENCE_LENGTH,
                                     EMBEDDING_DIM)
        lstm_model.summary()

        callbacks=[EarlyStopping(patience=2,
                                 monitor='val_loss')]

        history = lstm_model.fit(seq_array, 
                            next_word_array,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            epochs=NUM_EPOCHS,
                            callbacks=callbacks,
                            validation_split=0.1)

        logging.info("Saving model to {}".format(lstm_model_path))
        lstm_model.save(MODEL_DIR + "/" + 'lstm.h5')

    test_seq = TEST_SEED
    for word in range(TEST_NUM):
        test_vec = model.wv[test_seq[-SEQUENCE_LENGTH:]]
        test_vec = np.expand_dims(test_vec, 0)
        test_next = lstm_model.predict(test_vec)
        next_word = model.most_similar(positive=[test_next[0]], topn=1)
        test_seq.append(next_word[0][0])

    print(test_seq)

def read_input(file_list):
    for input_file in file_list:
        logging.info("reading file {0}...this may take a while".format(input_file))
        with open(input_file, 'r') as file:
            for i, line in enumerate(file):
                # do some pre-processing and return list of words for each review
                # text
                yield gensim.utils.simple_preprocess(line.split(":")[-1])

def get_gordon(file_list):
    for input_file in file_list:
        logging.info("reading file {0}...this may take a while".format(input_file))
        with open(input_file, 'r') as file:
            for i, line in enumerate(file):
                # do some pre-processing and return list of words for each review
                # text
                if line[:7] == "Gordon:":
                    yield gensim.utils.simple_preprocess(line.split(":")[-1])

def bidirectional_lstm_model(seq_length,
                             vocab_size):
    logging.info('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(RNN_SIZE, activation="relu"),
                            input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='logcosh',
                  optimizer=optimizer,
                  metrics=["acc"])
    logging.info("model built!")

    return model

if __name__ == '__main__':
    main()
