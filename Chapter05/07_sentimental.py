from keras import datasets, layers, models, preprocessing

MAX_LEN = 200
N_WORDS = 10000
DIM_EMBEDDING = 256
EPOCHS = 20
BATCH_SIZE = 500


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=N_WORDS)
    # max_len으로 시퀀스 패딩

    x_train = preprocessing.sequence.pad_sequences(x_train, MAX_LEN)
    x_test = preprocessing.sequence.pad_sequences(x_test, MAX_LEN)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    # model의 (batch, input_length) 크기의 정수 행렬을 입력으로 받는다
    # (input_length, dim_embedding)은 차원의 출력을 가진다.
    # 입력에서 최대 정수는 n_words(어휘 크기) 작거나 같아야한다.
    model = models.Sequential([
        layers.Embedding(N_WORDS, DIM_EMBEDDING, input_length=MAX_LEN),
        layers.Dropout(.3),
        layers.Conv1D(256, 3, padding='valid', activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.5),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

(x_train, y_train), (x_test, y_test) = load_data()
model = build_model()
model.summary()

print(x_train.shape, x_test.shape)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

score = model.fit(x_train, y_train, epochs = EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print('\nTest score', score[0])
print('Test accuracy', score[1])