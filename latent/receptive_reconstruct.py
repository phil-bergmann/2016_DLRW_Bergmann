
from autoencoder import autoencoder


def receptive_reconstruct():
    aec = autoencoder()
    aec.train()
    aec.train(sparse=0.001, image_save='autoencoderfilter_lambda0.001.png')
    aec.train(sparse=0.01, image_save='autoencoderfilter_lambda0.01.png')
    aec.train(sparse=0.1, image_save='autoencoderfilter_lambda0.1.png')
    aec.train(sparse=1.0, image_save='autoencoderfilter_lambda1.png')


if __name__ == '__main__':
    receptive_reconstruct()