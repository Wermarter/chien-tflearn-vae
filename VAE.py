import tensorflow as tf
import tflearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cv2

import tflearn.datasets.mnist as mnist
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)


class VAE(object):
    """Vanilla Variational AutoEncoder"""
    def __init__(self,
        log_dir = './vae_log',
        model_path = './vae_model.sav',
        learning_rate = 0.001,
        batch_size = 256,
        latent_dim = 2,
        binary_data = True, # 0-1, not thresholded
        img_shape = [28, 28, 1], # W, H, D. Must be a list()
        use_conv = False
        ):
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.binary_data = binary_data
        self.img_shape = img_shape
        self.use_conv = use_conv
        self.input_dim = img_shape if use_conv else [np.prod(img_shape)]
        self.full_graph = False
        self.model_path = model_path
        self._build_training_model()


    def conv_encode(self, input_data):
        conv1 = tflearn.conv_2d(input_data, 32, 5, activation='elu')
        pool1 = tflearn.max_pool_2d(conv1, 2)
        norm1 = tflearn.local_response_normalization(pool1)
        return tf.reshape(norm1, [self.curr_batch_size, 6272])

    def conv_decode(self, z_sampled):
        recon_activ = 'sigmoid' if self.binary_data else 'linear'
        fc1 = tflearn.fully_connected(z_sampled, 196, activation='elu')
        fc1 = tf.reshape(fc1, [-1, 14, 14, 1])
        deconv2 = tflearn.conv_2d_transpose(fc1, 1, 3, self.img_shape, 2, activation=recon_activ)
        return deconv2

    def fc_encode(self, input_data):
        net = tflearn.fully_connected(input_data, 300, activation='elu')
        net = tflearn.fully_connected(net, 100, activation='elu')
        return net

    def fc_decode(self, z_sampled):
        recon_activ = 'sigmoid' if self.binary_data else 'linear'
        net = tflearn.fully_connected(z_sampled, 100, activation='elu')
        net = tflearn.fully_connected(net, 300, activation='elu')
        net = tflearn.fully_connected(net, 784, activation=recon_activ)
        return net

    def _encode(self, input_data, is_training):
        with tf.variable_scope('Encoder', reuse=not is_training):
            if self.use_conv:
                net = self.conv_encode(input_data)
            else:
                net = self.fc_encode(input_data)
            z_mean = tflearn.fully_connected(net, self.latent_dim)
            z_std = tflearn.fully_connected(net, self.latent_dim)
        return z_mean, z_std
        
    def _decode(self, z_sampled, is_training):
        with tf.variable_scope('Decoder', reuse=not is_training):
            if self.use_conv:
                net = self.conv_decode(z_sampled)
            else:
                net = self.fc_decode(z_sampled)
        return net

    def _sample_z(self, z_mean, z_std):
        eps = tf.random_normal((self.curr_batch_size, self.latent_dim))
        return z_mean + tf.exp(z_std / 2) * eps

    def _compute_latent_loss(self, z_mean, z_std):
        latent_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
        latent_loss = -0.5 * tf.reduce_sum(latent_loss)
        return latent_loss

    def _compute_recon_loss(self, recon_data, input_data):
        recon_data = tf.reshape(recon_data, [self.curr_batch_size, -1])
        input_data = tf.reshape(input_data, [self.curr_batch_size, -1])
        if self.binary_data:
            loss = input_data*tf.log(1e-10+recon_data) + (1-input_data)*tf.log(1e-10+1-recon_data)
        else:
            loss = tflearn.objectives.mean_square(recon_data, input_data)
        return -tf.reduce_sum(loss)

    def _build_training_model(self):
        self.train_data = tflearn.input_data(shape=[None, *self.input_dim])
        self.curr_batch_size = tf.shape(self.train_data)[0]
        z_mean, z_std = self._encode(self.train_data, True)
        z_sampled = self._sample_z(z_mean, z_std)
        recon_data = self._decode(z_sampled, True)

        loss = self._compute_latent_loss(z_mean, z_std) + self._compute_recon_loss(recon_data, self.train_data)
        optimizer = tflearn.optimizers.Adam(self.learning_rate).get_tensor()
        trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, batch_size=self.batch_size, name='VAE_trainer')
        self.training_model = tflearn.Trainer(train_ops=trainop, tensorboard_dir=self.log_dir)

    def _build_full_graph(self):
        # Build generator model
        input_noise = tflearn.input_data(shape=[None, self.latent_dim], name='input_noise')
        decoded_noise = self._decode(input_noise, False)
        self.generator_model = tflearn.DNN(decoded_noise, session=self.training_model.session)

        # Build recognition model
        input_data = tflearn.input_data(shape=[None, *self.input_dim], name='input_data')
        self.curr_batch_size = tf.shape(input_data)[0]
        encoded_data = self._sample_z(*self._encode(input_data, False))
        self.recognition_model = tflearn.DNN(encoded_data, session=self.training_model.session)

        self.full_graph = True

    def fit(self, trainX, testX, n_epoch=100):
        n_train, n_test = trainX.shape[0], testX.shape[0]
        trainX = trainX.reshape((n_train, *self.input_dim))
        testX = testX.reshape((n_test, *self.input_dim))
        self.training_model.fit({self.train_data: trainX}, n_epoch, {self.train_data: testX}, run_id='VAE')

    def generate(self, input_noise=None, show_img=True):
        if not self.full_graph:
            self._build_full_graph()
        if input_noise is None:
            input_noise = np.random.normal(size=(1, self.latent_dim))
        else:
            input_noise = input_noise.reshape((-1, self.latent_dim))
        output = np.array(self.generator_model.predict({'input_noise': input_noise}))
        if not self.use_conv:
            output = output.reshape((-1, *self.img_shape))
        if show_img:
            self._imshow(output)
        return output

    def encode(self, input_data):
        if not self.full_graph:
            self._build_full_graph()
        input_data = input_data.reshape((-1, *self.input_dim))
        code = self.recognition_model.predict({'input_data': input_data})
        return np.array(code)

    def reconstruct(self, input_data, show_img=True):
        if not self.full_graph:
            self._build_full_graph()
        code = self.encode(input_data)
        recon = self.generate(code, show_img)
        return recon

    def format_img(self, img):
        img = np.array(img, np.float32)
        if np.prod(img.shape)==np.prod(self.img_shape):
            img = img.reshape((*self.img_shape))
        if self.img_shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def img_transition(self, A, B, step=10):
        enc_A = self.encode(A)[0]
        enc_B = self.encode(B)[0]
        trans_lst = list()
        for a, b in zip(enc_A, enc_B):
            trans_step = np.linspace(a, b, step)
            trans_lst.append(trans_step)
        trans_lst = np.array(trans_lst).T

        img_W, img_H, img_D = self.img_shape
        figure = np.ones((img_H, img_W*step, img_D))
        for i, trans_vec in enumerate(trans_lst):
            figure[:, i*img_W:(i+1)*img_W, :] = self.generate(trans_vec, False)
        self._imshow(figure)

    def show_2D_latent_space(self, graph_shape=(10, 10)):
        img_W, img_H, img_D = self.img_shape
        figure = np.ones((img_H*graph_shape[0], img_W*graph_shape[1], img_D))
        X = norm.ppf(np.linspace(0., 1., graph_shape[0]))
        Y = norm.ppf(np.linspace(0., 1., graph_shape[1]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                _recon = self.generate(np.array([[x, y]]), False)
                figure[i*img_H:(i+1)*img_H, j*img_W:(j+1)*img_W, :] = _recon
        self._imshow(figure)

    def _imshow(self, img):
        output_img = self.format_img(img)
        plt.figure()
        plt.imshow(output_img)
        plt.show()

    def save(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.training_model.save(model_path)

    def load(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.training_model.restore(model_path)
        self._build_full_graph()

def main():
    vae = VAE()
    vae.load()
    # vae.fit(trainX, testX, 10)
    # vae.save()
    vae.show_2D_latent_space()
    vae.img_transition(trainX[4], trainX[100])
    vae._imshow(trainX[4])
    vae.reconstruct(trainX[4])


if __name__=='__main__':
    main()