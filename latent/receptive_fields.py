from autoencoder import autoencoder


def receptive_fields():

    aec = autoencoder(n_hidden=1000)
    aec.train(image_save='autoencoderfilter_lambda0.png')
    aec.reconstruct_test(count=100, image_save='autoencoderrec_lambda0.png')

    aec.train(sparse=0.001, image_save='autoencoderfilter_lambda0.001.png')
    aec.reconstruct_test(count=100, image_save='autoencoderrec_lambda0.001.png')

    aec.train(sparse=0.01, image_save='autoencoderfilter_lambda0.01.png')
    aec.reconstruct_test(count=100, image_save='autoencoderrec_lambda0.01.png')

    aec.train(sparse=0.1, image_save='autoencoderfilter_lambda0.1.png')
    aec.reconstruct_test(count=100, image_save='autoencoderrec_lambda0.1.png')

    aec.train(sparse=1.0, image_save='autoencoderfilter_lambda1.png')
    aec.reconstruct_test(count=100, image_save='autoencoderrec_lambda1.png')

if __name__ == '__main__':
    receptive_fields()
