import tensorflow as tf
from keras import layers, models

# Define CNN for visual processing
cnn_model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten()
])
cnn_model.summary()

# Define the visual_model with proper input
image_input = layers.Input(shape=(224, 224, 3))
visual_model = cnn_model(image_input)

# TEXT

# Define the RNN model for text processing
question_input = layers.Input(shape=(100, ), dtype='int32')
embedding = layers.Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = layers.LSTM(256)(embedding)

# combine CNN RNN
merged = layers.concatenate([encoded_question, visual_model])
# 마지막에 밀집 망을 추가
output = layers.Dense(1000, activation='softmax')(merged)

# 병합된 모델 얻기
vqa_model = models.Model(inputs=[image_input, question_input], outputs=output)
vqa_model.summary()
