import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import pandas as pd

true_df = pd.read_csv('true.csv')
false_df = pd.read_csv('fake.csv')


true_df['label'] = 1
false_df['label'] = 0

df = pd.concat([true_df, false_df])


texts = df['text'].values
labels = df['label'].values


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


max_seq_length = 100  
sequences = pad_sequences(sequences, maxlen=max_seq_length)

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=0)

embedding_dim = 50

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_seq_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

predictions = model.predict(X_test)
predictions = [1 if p > 0.5 else 0 for p in predictions]

print("Doğruluk Oranı:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))