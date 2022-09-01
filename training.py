import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize tiap kata
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # ditambahkan ke list kelas/intents
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize dan mengecilkan tiap kata dan menghapus kata yang sama
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = umlah dokumen yang telah digabungkan antara patterns dengan intents
print (len(documents), "Jumlah dokumen yang telah digabungkan antara patterns dengan intents")
# classes = intents
print (len(classes), "Jumlah Intents", classes)
# words = Semua kata
print (len(words), "Kata yang sudah di processing", words)


pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))

# Membuat data training
training = []
# Membuat array kosong untuk output
output_empty = [0] * len(classes)
# training set, bag of words untuk tiap kata
for doc in documents:
    # inisiasi bag of words
    bag = []
    # list tiap kata yang sudah di tokenisasi
    pattern_words = doc[0]
    # lemmatize setiap kata, membuat kata dasarnya dengan tujuan untuk diberikan pada kata yang berhubungan
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # membuat bag of words dengan array 1 jika kata yang ditemukan sama dengan pattern saat ini
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # outputnya adalah '0' untuk tiap tag dan '1' untuk tag saat ini (untuk setiap pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# Acak tiap kata dan ubah ke bentuk array
random.shuffle(training)
training = np.array(training)
# membuat data train dan test, X untuk patterns dan Y untuk intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Data Training Telah Dibuat")


# Buat layers
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting data dan simpan modelnya
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("model created")