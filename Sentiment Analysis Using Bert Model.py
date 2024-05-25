import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow_datasets as tfds
def load_imdb_data():
    (train_data, test_data), info = tfds.load(
        'imdb_reviews',
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        as_supervised=True,
        with_info=True
    )
    return train_data, test_data, info
def preprocess_text(text):
    return tf.constant([text])
def build_model():
    text_input = Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name='preprocessing'
    )
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
        trainable=True,
        name='BERT_encoder'
    )
    outputs = encoder(encoder_inputs)
    pooled_output = outputs['pooled_output']
    dropout = Dropout(0.1)(pooled_output)
    output = Dense(1, activation='sigmoid', name='classifier')(dropout)

    model = Model(inputs=[text_input], outputs=[output])
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model
def train_model(model, train_data, test_data):
    train_data = train_data.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    test_data = test_data.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=3
    )
    return model, history
def analyze_sentiment(model, text):
    inputs = preprocess_text(text)
    prediction = model.predict(inputs)
    return "POSITIVE" if prediction[0][0] > 0.5 else "NEGATIVE", prediction[0][0]

if __name__ == "__main__":
    train_data, test_data, info = load_imdb_data()
    model = build_model()
    print(model.summary())
    model, history = train_model(model, train_data, test_data)
    text = input("Enter the text you want to analyze: ")
    label, score = analyze_sentiment(model, text)
    print(f"Label: {label}, Score: {score}")

