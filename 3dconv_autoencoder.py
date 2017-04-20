from keras.layers import Conv3D, MaxPool3D, UpSampling3D, Input
from conv3d_tied import Conv3D_tied
from keras.models import Model


class Stacked3DCAE(object):

    def __init__(self, input_shape, enc_conv_params, enc_maxpool_params, dec_deconv_params, dec_upsample_params):

        assert len(enc_conv_params) == len(enc_maxpool_params), 'Network should have as many conv-layers ' \
                                                                'as maxpool-layers'
        assert len(dec_deconv_params) == len(dec_upsample_params), 'Network should have as many deconv-layers ' \
                                                                   'as upsample layers'
        assert len(dec_deconv_params) == len(enc_conv_params), 'Number of encoding and decoding layers ' \
                                                               'has to be matched'
        self.input_shape = input_shape
        self.enc_conv_params = enc_conv_params
        self.enc_maxpool_params = enc_maxpool_params
        self.dec_deconv_params = dec_deconv_params
        self.dec_upsample_params = dec_upsample_params

        self.n_layers = len(self.enc_conv_params)
        self.layers = []
        self.encoding = None
        self.autoencoder = None

        self.build_network()

    def build_network(self):
        input_layer = Input(shape=self.input_shape, name='input_layer')

        # do encoding
        x = input_layer
        for i_layer in xrange(self.n_layers):
            conv_params = self.enc_conv_params[i_layer]
            maxpool_params = self.enc_maxpool_params[i_layer]
            x = self.build_layer_block_encoding(x, conv_params, maxpool_params, i=i_layer+1)

        encoding_output = x
        self.encoding = self.build_model(input_layer, encoding_output)

        # do decoding
        for i_layer in xrange(self.n_layers):
            deconv_params = self.dec_deconv_params[i_layer]
            upsample_params = self.dec_upsample_params[i_layer]

            # you need to inversely index the self.layers list
            x = self.build_layer_block_decoding(x, deconv_params, upsample_params, self.layers[-(i_layer+1)],
                                                i=i_layer+1)

        self.autoencoder = self.build_model(input_layer, x)

    @staticmethod
    def build_model(input_layer, encoding):
        return Model(input_layer, encoding)

    def build_layer_block_encoding(self, input_layer, conv_params, max_pool_params, i=1):
        encoding = Conv3D(name='enc_conv{}'.format(i), **conv_params)
        self.layers.append(encoding)
        x = encoding(input_layer)
        return MaxPool3D(name='enc_maxpool{}'.format(i), **max_pool_params)(x)

    @staticmethod
    def build_layer_block_decoding(input_layer, deconv_params, upsample_params, tied_layer, i=1):
        x = UpSampling3D(name='dec_upsample{}'.format(i), **upsample_params)(input_layer)
        return Conv3D_tied(name='dec_deconv{}'.format(i), tied_to=tied_layer, **deconv_params)(x)

    def summary(self):
        self.autoencoder.summary()

    def compile(self):
        assert self.autoencoder.input_shape == self.autoencoder.output_shape, ''
        self.autoencoder.compile('adam', loss='mean_squared_error')


def build_stacked_cae():
    num_output = 1

    input_shape = (96, 120, 96, num_output)

    enc_conv_params = [
        {'filters': 60, 'kernel_size': 7, 'padding': 'same', 'activation': 'relu'},
        {'filters': 40, 'kernel_size': 5, 'padding': 'same', 'activation': 'relu'},
        {'filters': 20, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}
    ]

    enc_maxpool_params = [
        {'pool_size': (2, 2, 2), 'padding': 'same'},
        {'pool_size': (2, 2, 2), 'padding': 'same'},
        {'pool_size': (2, 2, 2), 'padding': 'same'}
    ]

    dec_deconv_params = [
        {'filters': 40, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
        {'filters': 60, 'kernel_size': 5, 'padding': 'same', 'activation': 'relu'},
        {'filters': num_output, 'kernel_size': 7, 'padding': 'same', 'activation': 'relu'}
    ]

    dec_upsample_params = [
        {'size': (2, 2, 2)},
        {'size': (2, 2, 2)},
        {'size': (2, 2, 2)}
    ]
    cae3d = Stacked3DCAE(input_shape, enc_conv_params, enc_maxpool_params, dec_deconv_params, dec_upsample_params)
    cae3d.compile()
    cae3d.summary()
    return cae3d


if __name__ == '__main__':
    cae = build_stacked_cae()
