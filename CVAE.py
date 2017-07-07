import VAE
import tensorflow as tf
import tflearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cv2

import tflearn.datasets.mnist as mnist
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

class CVAE(VAE.VAE):
    """Conditional Variational AutoEncoder"""
    def __init__(self,
        log_dir = './cvae_log',
        model_path = './cvae_model.sav',
        learning_rate = 0.001,
        batch_size = 256,
        latent_dim = 2,
        binary_data = True, # 0-1, not thresholded
        img_shape = [28, 28, 1], # W, H, D
        ref_dim = 10
        ):
        self.ref_dim = ref_dim
        super(CVAE, self).__init__(
            log_dir,
            model_path,
            learning_rate,
            batch_size,
            latent_dim,
            binary_data,
            img_shape,
            use_conv=False) # My GPU is not so strong + My knowledge of convolution is not so deep -> No more ConvLayer


    def _build_training_model(self):
        self.train_data = tflearn.input_data(shape=[None, *self.input_dim], name='train_data')
        self.train_data_ref = tflearn.input_data(shape=[None, self.ref_dim], name='train_data_ref')
        self.curr_batch_size = tf.shape(self.train_data)[0]
        cmb_train_data = tf.concat([self.train_data, self.train_data_ref], 1)
        z_mean, z_std = self._encode(cmb_train_data, True)
        z_sampled = self._sample_z(z_mean, z_std)
        cmb_z_ref = tf.concat([z_sampled, self.train_data_ref], 1)
        recon_data = self._decode(cmb_z_ref, True)

        loss = self._compute_latent_loss(z_mean, z_std) + self._compute_recon_loss(recon_data, self.train_data)
        optimizer = tflearn.optimizers.Adam(self.learning_rate).get_tensor()
        trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, batch_size=self.batch_size, name='VAE_trainer')
        self.training_model = tflearn.Trainer(train_ops=trainop, tensorboard_dir=self.log_dir)

    def _build_full_graph(self):
        # Build generator model
        input_noise = tflearn.input_data(shape=[None, self.latent_dim], name='input_noise')
        noise_ref = tflearn.input_data(shape=[None, self.ref_dim], name='noise_ref')
        cmb_noise = tf.concat([input_noise, noise_ref], 1)
        decoded_noise = self._decode(cmb_noise, False)
        self.generator_model = tflearn.DNN(decoded_noise, session=self.training_model.session)

        # Build recognition model
        input_data = tflearn.input_data(shape=[None, *self.input_dim], name='input_data')
        input_ref = tflearn.input_data(shape=[None, self.ref_dim], name='input_ref')
        self.curr_batch_size = tf.shape(input_data)[0]
        cmb_input = tf.concat([input_data, input_ref], 1)
        encoded_data = self._sample_z(*self._encode(cmb_input, False))
        self.recognition_model = tflearn.DNN(encoded_data, session=self.training_model.session)

        self.full_graph = True

    def fit(self, trainX, trainY, testX, testY, n_epoch=100):
        n_train, n_test = trainX.shape[0], testX.shape[0]
        trainX = trainX.reshape((n_train, *self.input_dim))
        testX = testX.reshape((n_test, *self.input_dim))
        self.training_model.fit({self.train_data: trainX, self.train_data_ref: trainY}, n_epoch, 
            {self.train_data: testX, self.train_data_ref: testY}, run_id='CVAE')

    def generate(self, noise_ref, input_noise=None, show_img=True):
        if not self.full_graph:
            self._build_full_graph()
        if input_noise is None:
            input_noise = np.random.normal(size=(1, self.latent_dim))
        else:
            input_noise = np.array(input_noise).reshape((-1, self.latent_dim))
        noise_ref = noise_ref.reshape((-1, self.ref_dim))
        output = np.array(self.generator_model.predict({'input_noise': input_noise, 'noise_ref': noise_ref}))
        if not self.use_conv:
            output = output.reshape((-1, *self.img_shape))
        if show_img:
            self._imshow(output)
        return output

    def encode(self, input_data, input_ref):
        if not self.full_graph:
            self._build_full_graph()
        input_data = input_data.reshape((-1, *self.input_dim))
        input_ref = input_ref.reshape((-1, self.ref_dim))
        return self.recognition_model.predict({'input_data': input_data, 'input_ref': input_ref})

    def reconstruct(self, input_data, input_ref, show_img=True):
        if not self.full_graph:
            self._build_full_graph()
        code = self.encode(input_data, input_ref)
        recon = self.generate(input_ref, code, show_img)
        return recon



if __name__=='__main__':
    cvae = CVAE()
    cvae.load()
    # cvae.fit(trainX, trainY, testX, testY, 10)
    # cvae.save()

    label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(10):
        cvae.generate(np.roll(label, i))
        # cvae.generate(label)
        pass

    cvae._imshow(trainX[4])
    cvae.reconstruct(trainX[4], trainY[4])

