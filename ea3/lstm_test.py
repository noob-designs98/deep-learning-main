import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import tensorflowjs as tfjs
import json
#read dataset
import os

count = 0
with tf.device("/device:GPU:0"):
    texts = []

#Preprocessing ist indentisch zu feed_forward
    for subdir, dirs, files in os.walk("emails"):
        for file in files:
            path = os.path.join(subdir, file)
            print(f"Reading: {path}")
            with open(path) as f:
                texts.append(f.read())
            count += 1
            if count >= 1000:
                break

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for text in texts:
        for line in text.split('\n'):
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

    #Berechne längste Sequenz
    max_sequence_len = max([len(seq) for seq in input_sequences])
    #Erzeuge eingabe Sequenzen
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))
    #Das Model Besitzt ein Embedding Layer, ein LSTM Layer mit 150 units und ein FC Layer
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=20, verbose=1)

    # Eingabe text für Word prediction
    seed_text = "Hallo"
    # Anzahl an vorhergsagten Wörtern
    next_words = 20

    for _ in range(next_words):
        #Konvertiere Eingabe Text zu Tokens
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #Sage nächstes Wort für aktuellen seed_text vor. Gleiche Logik wie bei feed_forward
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        # get predict words
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        #Hänge vorhergesagtes word an seed_text an und sage, falls noch nötig, das nächstes voraus
        seed_text += " " + output_word


    print(seed_text)

#Speicher Model und Vokabular
    tfjs.converters.save_keras_model(model, "output_lstm")

with open("lstm_dict.json", "w") as lstm_dict:
    lstm_dict.write(
        json.dumps(tokenizer.word_index)
    )
