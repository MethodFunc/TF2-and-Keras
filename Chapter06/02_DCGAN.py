import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, UpSampling2D, Conv2D, Flatten, Dropout, Reshape, BatchNormalization, Activation, Input, \
    LeakyReLU, ZeroPadding2D
from keras.optimizers import Adam

SEED = 2020
tf.random.set_seed(SEED)
np.random.seed(SEED)

class DCGAN():
    def __init__(self, rows, cols, channels, z = 10):
        # 입력 형태
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = z
        optimizer = Adam(0.0002, 0.5)

        # 판별기 구축과 컴파일
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # 생성기 구축
        self.generator = self.build_generator()
        # 생성기는 노이즈를 입력을 받아 이미지 생성
        z = Input(shape=(self.latent_dim, ))
        img= self.generator(z)

        # 판별기는 생성된 이미지를 입력으로 받아 진위 여부를 판단.
        valid = self.discriminator(img)

        #결합된 모델(생성기와 판별기를 쌓음)
        # 생성기를 훈련해 판별기를 속임
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential([
            Dense(128 * 7 * 7, activation='relu', input_dim=self.latent_dim),
            Reshape((7, 7, 128)),
            UpSampling2D(),
            Conv2D(128, 3, padding='same'),
            BatchNormalization(momentum=0.8),
            Activation('relu'),
            UpSampling2D(),
            Conv2D(64, 3, padding='same'),
            BatchNormalization(momentum=0.8),
            Activation('relu'),
            Conv2D(self.channels, kernel_size=3, padding='same'),
            Activation('tanh')
        ])

        model.summary()
        noise = Input(shape=(self.latent_dim))
        img = model(noise)

        return Model(noise, img)


    def build_discriminator(self):
        model = Sequential([
            Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'),
            LeakyReLU(0.2),
            Dropout(0.25),

            Conv2D(64, kernel_size=3, strides=2, padding='same'),
            ZeroPadding2D(padding=((0, 1), (0, 1))),
            BatchNormalization(momentum=0.8),
            LeakyReLU(0.2),
            Dropout(0.25),

            Conv2D(128, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            LeakyReLU(0.2),
            Dropout(0.25),

            Conv2D(256, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            LeakyReLU(0.2),
            Dropout(0.25),

            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        model.summary()
        img  = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size = 128, save_interval = 50):
        (x_train, _), (_, _) = mnist.load_data()
        # [-1 , 1] Normalize
        x_train = x_train / 127.5 - 1
        x_train = np.expand_dims(x_train, axis=3)

        # 적대적 참의 정의
        valid  = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # 임의로 이미지 반을 선택
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # 판별기 훈련 (진짜는 1, 가짜는 0)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig('../result/images/dcgan_mnist_%d.png'% epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN(28, 28, 1)
    dcgan.train(epochs=5000, batch_size=32, save_interval=50)