import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # token hóa từng từ
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # thêm tài liệu vào kho văn bản
        documents.append((w, intent['tag']))

        # thêm vào danh sách classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize và giảm từng từ và loại bỏ các từ trùng lặp
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sắp xếp classes
classes = sorted(list(set(classes)))
# documents = sự kết hợp giữa các mẫu và ý định
print(len(documents), "documents")
# classes =  ý định intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# create our training data
training = []
# create an empty array for our output .tạo một mảng trống cho đầu ra của chúng ta
output_empty = [0] * len(classes)
# training set, bag of words for each sentence.tập huấn luyện, túi từ cho mỗi câu
for doc in documents:
    # initialize our bag of words.# khởi tạo túi từ của chúng tôi
    bag = []
    # list of tokenized words for the pattern.# danh sách các từ được mã hóa cho mẫu
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words Hiển thị các ừ liên quan
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("model created")