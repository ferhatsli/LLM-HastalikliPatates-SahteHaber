import tensorflow as tf
import numpy as np
import pandas as pd

def load_data(fake_csv, true_csv):
    fake_df = pd.read_csv(fake_csv)
    true_df = pd.read_csv(true_csv)

    data = pd.concat([fake_df, true_df], ignore_index=True)

    return data

def preprocess_data(data):
    data['text'] = data['text'].str.lower()
    data['subject'] = data['subject'].str.lower()

    return data

def create_dataset(data):
    x_data = data['text'].tolist()
    y_data = data['subject'].tolist()

    return x_data, y_data

def train_model(x_data, y_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=len(set(x_data)), output_dim=128),
        tf.keras.layers.LSTM(units=128),
        tf.keras.layers.Dense(units=len(set(y_data)))
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_data, y_data, epochs=10)

    return model

def generate_fake_news(model, input_text):

    input_chars = list(input_text)


    input_chars.insert(0, '<start>')


    prediction_sequence = np.zeros(len(input_chars), dtype=np.int32)


    prediction_sequence[0] = input_chars[0]


    for i in range(1, len(input_chars)):

        prediction = model.predict(prediction_sequence[:i])


        prediction_sequence[i] = prediction[0][0]


    prediction_chars = [chr(c) for c in prediction_sequence]


    return ''.join(prediction_chars[1:])




data = load_data('fake.csv', 'true.csv')


data = preprocess_data(data)


x_data, y_data = create_dataset(data)


model = train_model(x_data, y_data)


input_text = 'Türkiye'


fake_news = generate_fake_news(model, input_text)


print(fake_news)