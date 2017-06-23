import tensorflow as tf
import tflearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import tflearn.datasets.mnist as mnist
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

TENSORBOARD_DIR='./logs/vae'

class VAE(object):
    
    def __init__(self,
        learning_rate = 0.001,
        batch_size = 256,
        input_dim = 784,
        latent_dim = 2,
        binary_data = True
        ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.binary_data = binary_data
        self._build_graph()

    def _encode(self, input_data, is_training):
        fc1 = tflearn.fully_connected(input_data, self.input_dim//2, activation='elu', scope='enc_fc1', reuse=not is_training)
        fc2 = tflearn.fully_connected(fc1, self.input_dim//4, activation='elu', scope='enc_fc2', reuse=not is_training)
        z_mean = tflearn.fully_connected(fc2, self.latent_dim, scope='enc_z_mean', reuse=not is_training)
        z_std = tflearn.fully_connected(fc2, self.latent_dim, scope='enc_z_std', reuse=not is_training)
        return z_mean, z_std

    def _sample_z(self, z_mean, z_std):
        batch_size = tf.shape(z_mean)[0]
        eps = tf.random_normal((batch_size, self.latent_dim))
        return z_mean + tf.exp(z_std / 2) * eps

    def _decode(self, z_sampled, is_training):
        recon_activ = 'sigmoid' if self.binary_data else 'linear'
        fc1 = tflearn.fully_connected(z_sampled, self.input_dim//4, activation='elu', scope='dec_fc1', reuse=not is_training)
        fc2 = tflearn.fully_connected(fc1, self.input_dim//2, activation='elu', scope='dec_fc2', reuse=not is_training)
        recon = tflearn.fully_connected(fc2, self.input_dim, activation=recon_activ, scope='dec_recon', reuse=not is_training)
        return recon

    def _compute_latent_loss(self, z_mean, z_std):
        latent_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
        latent_loss = -0.5 * tf.reduce_sum(latent_loss)
        return latent_loss

    def _compute_recon_loss(self, recon_data, input_data):
        if self.binary_data:
            loss = input_data*tf.log(1e-10+recon_data) + (1-input_data)*tf.log(1e-10+1-recon_data)
        else:
            loss = tflearn.objectives.mean_square(recon_data, input_data)
        return -tf.reduce_sum(loss)

    def _build_graph(self):
        # Build training model
        self.train_data = tflearn.input_data(shape=[None, self.input_dim], name='train_data')
        z_mean, z_std = self._encode(self.train_data, True)
        z_sampled = self._sample_z(z_mean, z_std)
        recon_data = self._decode(z_sampled, True)

        loss = self._compute_latent_loss(z_mean, z_std) + self._compute_recon_loss(recon_data, self.train_data)
        optimizer = tflearn.optimizers.Adam(self.learning_rate).get_tensor()
        trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, batch_size=self.batch_size, name='VAE_trainer')
        self.training_model = tflearn.Trainer(train_ops=trainop, tensorboard_dir=TENSORBOARD_DIR)

        # Build generator model
        self.input_noise = tflearn.input_data(shape=[1, self.latent_dim], name='input_noise')
        decoded_noise = self._decode(self.input_noise, False)
        self.generator_model = tflearn.DNN(decoded_noise, session=self.training_model.session)

        # Build recognition model
        self.input_data = tflearn.input_data(shape=[1, self.input_dim], name='input_data')
        encoded_data = self._sample_z(*self._encode(self.input_data, False))
        self.recognition_model = tflearn.DNN(encoded_data, session=self.training_model.session)

    def fit(self, X, testX, n_epoch=100):
        self.training_model.fit({self.train_data: X}    , n_epoch, {self.train_data: testX}, run_id='VAE')

    def generate(self, input_noise=None):
        if input_noise is None:
            input_noise = np.random.normal(size=(1, self.latent_dim))
        else:
            input_noise = np.array(input_noise).ravel()
        return self.generator_model.predict({'input_noise': input_noise.reshape((1, 1, max(input_noise.shape)))})

    def encode(self, input_data):
        input_data = np.array(input_data)
        return self.recognition_model.predict({'input_data': input_data.reshape((1, 1, max(input_data.shape)))})

    def MNIST_transition(self, A, B, step=10):
        enc_A = self.encode(A)[0]
        enc_B = self.encode(B)[0]
        trans_lst = list()
        for a, b in zip(enc_A, enc_B):
            trans_step = np.linspace(a, b, step)
            trans_lst.append(trans_step)
        trans_lst = np.array(trans_lst).T
        figure = np.ones((28, 28*step))
        for i, trans_vec in enumerate(trans_lst):
            figure[..., i*28:(i+1)*28] = self.generate(trans_vec).reshape(28, 28)
        plt.figure(figsize=(10/step, 10))
        plt.imshow(figure)
        plt.show()
        
    def MNIST_latent_space(self, graph_shape=(30, 30)):
        figure = np.ones((28*graph_shape[0], 28*graph_shape[1]))
        X = norm.ppf(np.linspace(0., 1., graph_shape[0]))
        Y = norm.ppf(np.linspace(0., 1., graph_shape[1]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                _recon = self.generate(np.array([[x, y]]))
                figure[i*28:(i+1)*28, j*28:(j+1)*28] = np.array(_recon).reshape((28, 28))
        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()

    def save(self):
        try:
            self.training_model.save('training_model.sav')
            self.generator_model.save('generator_model.sav')
            self.recognition_model.save('recognition_model.sav')
        except:
            # Some are used, some are not used
            pass

    def load(self):
        try:
            self.training_model.load('training_model.sav')
            self.generator_model.load('generator_model.sav')
            self.recognition_model.load('recognition_model.sav')
        except:
            pass

def main():
    vae = VAE()
    vae.fit(trainX, testX)
    vae.MNIST_latent_space()
    vae.MNIST_transition(trainX[4], trainX[100])
    vae.MNIST_transition(testX[4], testX[100])
    vae.save()


if __name__=='__main__':
    main()