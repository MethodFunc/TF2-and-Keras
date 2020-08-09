import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Dropout, Input
from keras import initializers
from keras.optimizers import Adam
from keras.datasets import mnist

randomDim = 10
np.random.seed(1000)

(x_train, _), (_, _) = mnist.load_data()
# [-1 , 1] Normalize
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(-1, 784)

# 생성자 정의
generator = Sequential([
    Dense(256, input_dim=randomDim),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(784, activation='tanh')
])

# 판별자 정의
discriminator = Sequential([
    Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(512),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(256),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
adam = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# 네트워크 합치기(생성자, 판별자)
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []


# 각 배치에서 손실 도식화
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../result/images/gan_loss_epoch_%d.png' % epoch)
    print('%d plotLoss save complete' % epoch)


# 생성된 MNIST 이미지 나열
def saveGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')

        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../result/images/gan_generated_image_epoch_%d.png' % epoch)
    print('%d GeneratedImage save complete' % epoch)


def train(epochs=1, batchsize=128):
    batchCount = int(x_train.shape[0] / batchsize)
    print('Epochs:', epochs)
    print('Batch size:', batchsize)
    print('Batches per epochs:', batchCount)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in range(batchCount):
            # 랜덤 입력 노이즈와 이미지를 얻는다.
            noise = np.random.normal(0, 1, size=[batchsize, randomDim])
            imageBatch = x_train[np.random.randint(0, x_train.shape[0], size=batchsize)]

            # 가짜 MNIST이미지 생성
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # 생성된 것과 실제 이미지의 레이블
            yDis = np.zeros(2 * batchsize)
            # 편파적 레이블 평활화
            yDis[:batchsize] = 0.9

            # 판별기 훈련
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # 생성기 훈련
            noise = np.random.normal(0, 1, size=[batchsize, randomDim])
            yGen = np.ones(batchsize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # 이 에폭의 최근 배치에서의 손실을 저장
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 20 == 0:
            saveGeneratedImages(e)

    plotLoss(e)

if __name__ == "__main__":
    train(200, 128)
