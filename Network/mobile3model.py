from keras import backend as keras_backend
from keras import layers
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, GlobalAveragePooling2D
from keras.layers import ConvLSTM2D, TimeDistributed, Activation, Add, DepthwiseConv2D, Reshape
from keras.models import Model

T = 1
SQUEEZE_AND_EXCITE = True

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = TimeDistributed(GlobalAveragePooling2D())(init)
    se = TimeDistributed(Reshape(se_shape))(se)
    se = TimeDistributed(Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False))(se)
    se = TimeDistributed(Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False))(se)

    # if keras_backend.image_data_format() == 'channels_first':
    #     se = TimeDistributed(Permute((3, 1, 2)))(se)

    x = layers.multiply([init, se])
    return x

### Code from Liums repo ###
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return keras_backend.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if keras_backend.image_data_format() == 'channels_first' else -1

    x = TimeDistributed(Conv2D(filters, kernel, padding='same', strides=strides))(inputs)
    x = TimeDistributed(BatchNormalization(axis=channel_axis))(x)
    return TimeDistributed(Activation(relu6))(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, squeeze, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if keras_backend.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = keras_backend.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = TimeDistributed(DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization(axis=channel_axis))(x)
    x = TimeDistributed(Activation(relu6))(x)

    if squeeze:
        x = squeeze_excite_block(x)

    x = TimeDistributed(Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same'))(x)
    x = TimeDistributed(BatchNormalization(axis=channel_axis))(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n, squeeze):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides, squeeze)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, squeeze, True)

    return x


def decoder_network(inputs):
    x_e = Conv2DTranspose(512, 4, strides=1, padding='same', activation=keras_backend.relu)(inputs)  # 8 x 8 x 512

    x_e = Conv2DTranspose(256, 4, strides=2, padding='same', activation=keras_backend.relu)(x_e)  # 16 x 16 x 256
#    x_e = Conv2DTranspose(256, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 16 x 16 x 256
#    x_e = Conv2DTranspose(256, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 16 x 16 x 256
    
    x_e = Conv2DTranspose(128, 4, strides=2, padding='same', activation=keras_backend.relu)(x_e)  # 32 x 32 x 128
#    x_e = Conv2DTranspose(128, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 32 x 32 x 128
#    x_e = Conv2DTranspose(128, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 32 x 32 x 128
    
    x_e = Conv2DTranspose(64, 4, strides=2, padding='same', activation=keras_backend.relu)(x_e)  # 64 x 64 x 64
#    x_e = Conv2DTranspose(64, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 64 x 64 x 64
#    x_e = Conv2DTranspose(64, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 64 x 64 x 64

    x_e = Conv2DTranspose(32, 4, strides=2, padding='same', activation=keras_backend.relu)(x_e)  # 128 x 128 x 32
#    x_e = Conv2DTranspose(32, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 128 x 128 x 32
    
    x_e = Conv2DTranspose(16, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 256 x 256 x 16
#    x_e = Conv2DTranspose(16, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 256 x 256 x 16

#    x_e = Conv2DTranspose(3, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 256 x 256 x 3
    x_e = Conv2DTranspose(8, 4, strides=1, padding='same', activation=keras_backend.relu)(x_e)  # 256 x 256 x 3
    x_e = Conv2DTranspose(3, 4, strides=1, padding='same', activation=keras_backend.tanh)(x_e)

    return x_e


def get_keras_model(input_shape, N=2**14, alpha=1.0, squeeze=False):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)

    first_filters = _make_divisible(32 * alpha, 8)
#    x_e = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))
    x_e = ConvLSTM2D(first_filters, kernel_size=(3,3), strides=(2, 2), padding='same', return_sequences=True)(inputs)


    # x_e = _inverted_residual_block(x_e, base_filters, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    # x_e = _inverted_residual_block(x_e, base_filters + 8, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    # x_e = _inverted_residual_block(x_e, base_filters + 8*2, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    # x_e = _inverted_residual_block(x_e, base_filters + 8*3, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    # x_e = _inverted_residual_block(x_e, base_filters + 8*4, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    # x_e = _inverted_residual_block(x_e, base_filters + 8*5, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    # x_e = _inverted_residual_block(x_e, base_filters + 8*6, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    x_e = _inverted_residual_block(x_e, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1, squeeze=squeeze)
#    x_e = _inverted_residual_block(x_e, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2, squeeze=squeeze)
    x_e = ConvLSTM2D(24, kernel_size=(3,3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
#    x_e = _inverted_residual_block(x_e, 32, (3, 3), t=6, alpha=alpha, strides=2, n=2, squeeze=squeeze) #n=3
    x_e = ConvLSTM2D(32, kernel_size=(3,3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
#    x_e = _inverted_residual_block(x_e, 64, (3, 3), t=6, alpha=alpha, strides=2, n=3, squeeze=squeeze) #n=4
    x_e = ConvLSTM2D(64, kernel_size=(3,3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
#    x_e = _inverted_residual_block(x_e, 96, (3, 3), t=6, alpha=alpha, strides=1, n=2, squeeze=squeeze) #n=3
    x_e = ConvLSTM2D(96, kernel_size=(3,3), strides=(1, 1), padding='same', return_sequences=True)(x_e)
    x_e = _inverted_residual_block(x_e, 160, (3, 3), t=6, alpha=alpha, strides=2, n=2, squeeze=squeeze) #n=3
    x_e = _inverted_residual_block(x_e, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1, squeeze=squeeze)
    
    if alpha > 1.0:
        last_filters = _make_divisible(512 * alpha, 8)
    else:
        last_filters = 512
    
#    x_e = _conv_block(x_e, last_filters, (1, 1), strides=(1, 1))

    x_e = ConvLSTM2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x_e)

    x_e = ConvLSTM2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=False)(x_e)
    
    x_e = decoder_network(x_e)

    x_e = Reshape((N, 3))(x_e)

    model = Model(inputs, x_e)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    return model


if __name__=='__main__':
    model = get_keras_model(8, 3, (5, 256, 256, 3))

    print(model.summary())
