import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.losses import CategoricalCrossentropy
import tensorflowjs as tfjs
#read dataset
import os
import json

#Generiert für alle Sequenzen in sequences eine Liste an Sequenten die
#auf die Länge length mit null gepaddet bzw auf die Länge length geschnitten werden
def get_sequences_of_len(sequences, length):
    result = []
    for sequence in sequences:
        if len(sequence) > length:
            sequence = sequence[len(sequence) - length:]
        elif len(sequence) < length:
            sequence = [0] * (length - len(sequence)) + sequence

        result.append(sequence)
    return result

#Generiert aus den Sequenten in sequnces das Training set
#Für die Eingabe Sequent [22, 35, 15, 89] ist das erzeugte 
#x: [22, 35, 15] und das erwartete y 89
def generate_train_set(sequences, total_words):
    x = []
    y = []
    for seq in sequences:
        x.append(seq[:-1])
        tmp = [0] * total_words
        tmp[seq[-1]] = 1
        y.append(tmp)

    return x, y 



count = 0
with tf.device("/device:GPU:0"):
    texts = []

#Gehe durch die ersten 100 mails
    for subdir, dirs, files in os.walk("emails"):
        for file in files:
            path = os.path.join(subdir, file)
            print(f"Reading: {path}")
            with open(path) as f:
                texts.append(f.read())
            count += 1
            if count >= 100:
                break

    #Tokenizer
    tokenizer = Tokenizer()
    #fit
    tokenizer.fit_on_texts(texts)
    #Berechnte anzahl an Worten
    total_words = len(tokenizer.word_index) + 1

    #Erzeuge ngrams
    input_sequences = []
    for text in texts:
        for line in text.split('\n'):
            #Erzeuge token liste pro Zeile
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
    
#Seq_size gibt die Länge des Betrachteten Zeitfensters an
seq_size = 10 # Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.
seq = get_sequences_of_len(input_sequences, seq_size)
x,y = generate_train_set(seq, total_words)



print('Build deep model...')
#Baue und trainiere Model
model = Sequential()
#Input dimension ist vector mit der Länge des Betrachteten Zeitfensters
model.add(Dense(64, input_dim=seq_size - 1, activation='relu')) #12
model.add(Dense(32, activation='relu'))  #8
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x, y, epochs=100)
print(model.summary()) 

#Ein simpeler Test, erzeuge einen Beispielsatz, und wandele ihn in einer zu verarbeitende Sequenz um
token_list = tokenizer.texts_to_sequences(["was ist dein traum"])[0]
predict_seq = get_sequences_of_len([token_list], seq_size - 1)

#model.predict gibt einen vector der Länge total_words zurück. Für jedes Wort
#im Vokabular ist die Wahrscheinlichkeit für sein auftreten enthalten.
#np.argmax(model.predict) gibt den Index des wahrscheinlichsten Wortes
predicted = np.argmax(model.predict(predict_seq), axis=-1)
#Suche nach Wert des vorgesagtem Indexes
for word, index in tokenizer.word_index.items():
    if index == predicted:
        print(f"Predicted {word}")
        break

#Speichere Modell und Vokabular für die Webseite
tfjs.converters.save_keras_model(model, "output_feed_forward")

with open("feed_forward_dict.json", "w") as feed_forward_dict:
    # magic happens here to make it pretty-printed
    feed_forward_dict.write(
        json.dumps(tokenizer.word_index)
    )
