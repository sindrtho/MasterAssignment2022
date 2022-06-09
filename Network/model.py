from keras import backend as keras_backend
from keras import layers
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten
from keras.models import Model
from keras.layers import ConvLSTM2D, TimeDistributed

T = 1

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = TimeDistributed(GlobalAveragePooling2D())(init)
    se = TimeDistributed(Reshape(se_shape))(se)
    se = TimeDistributed(Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False))(se)
    se = TimeDistributed(Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False))(se)

    # if K.image_data_format() == 'channels_first':
    #     se = TimeDistributed(Permute((3, 1, 2)))(se)

    x = layers.multiply([init, se])
    return x

# # from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64
if T == 0:
    def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        # down-sampling is performed with a stride of 2
        y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)

        y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = LeakyReLU()(y)

        return y
elif T == 1:
    def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=True):
        shortcut = y

        # down-sampling is performed with a stride of 2
        y = TimeDistributed(Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(LeakyReLU())(y)

        y = squeeze_excite_block(y)

        y = TimeDistributed(Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = TimeDistributed(Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same'))(shortcut)
            shortcut = TimeDistributed(BatchNormalization())(shortcut)

        y = layers.add([shortcut, y])
        y = TimeDistributed(LeakyReLU())(y)

        return y


# Original network as proposed by Ola Lium
def get_keras_model(base_filters, input_shape, N=27508):  # 30247 is the lowest amount of points available across all given ground truths and is the number of points sampled from all other ground truths.
        inputs = Input(shape=input_shape)

        # encoder
        x_e =TimeDistributed( Conv2D(base_filters, 4, strides=1, padding='same'))(inputs)  # 256 x 256 x 16

        x_e = residual_block(inputs, base_filters * 2, _strides=(2, 2))  # 128 x 128 x 32
        x_e = residual_block(x_e, base_filters * 2, _strides=(1, 1))  # 128 x 128 x 32

        x_e = residual_block(x_e, base_filters * 4, _strides=(2, 2))  # 64 x 64 x 64
        x_e = residual_block(x_e, base_filters * 4, _strides=(1, 1))  # 64 x 64 x 64

        x_e = residual_block(x_e, base_filters * 8, _strides=(2, 2))  # 32 x 32 x 128
        x_e = residual_block(x_e, base_filters * 8, _strides=(1, 1))  # 32 x 32 x 128

        x_e = residual_block(x_e, base_filters * 16, _strides=(2, 2))  # 16 x 16 x 256
        x_e = residual_block(x_e, base_filters * 16, _strides=(1, 1))  # 16 x 16 x 256

        x_e = residual_block(x_e, base_filters * 32, _strides=(2, 2))  # 8 x 8 x 512
        x_e = residual_block(x_e, base_filters * 32, _strides=(1, 1))  # 8 x 8 x 512

        #x_e = ConvLSTM2D(base_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
        # x_e = ConvLSTM2D(base_filters * 2, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x_e)

        #x_e = ConvLSTM2D(base_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
        # x_e = ConvLSTM2D(base_filters * 4, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x_e)

        #x_e = ConvLSTM2D(base_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
        # x_e = ConvLSTM2D(base_filters * 8, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x_e)

        #x_e = ConvLSTM2D(base_filters * 16, kernel_size=(3, 3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
        #x_e = ConvLSTM2D(base_filters * 16, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x_e)

        #x_e = ConvLSTM2D(base_filters * 32, kernel_size=(3, 3), strides=(2, 2), padding='same', return_sequences=True)(x_e)
        x_e = ConvLSTM2D(base_filters * 32, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=True)(x_e)

        x_e = ConvLSTM2D(base_filters * 32, kernel_size=(3, 3), strides=(1, 1), padding='same', return_sequences=False)(x_e)

        # decoder
        x_e = Conv2DTranspose(base_filters * 16, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 8 x 8 x 512

        x_e = Conv2DTranspose(base_filters * 8, 4, strides=(2, 2), padding='same', activation=keras_backend.relu)(x_e)  # 16 x 16 x 256
        x_e = Conv2DTranspose(base_filters * 8, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 16 x 16 x 256
        #x_e = Conv2DTranspose(base_filters * 8, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 16 x 16 x 256

        x_e = Conv2DTranspose(base_filters * 4, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 32 x 32 x 128
        x_e = Conv2DTranspose(base_filters * 4, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 32 x 32 x 128
        #x_e = Conv2DTranspose(base_filters * 4, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 32 x 32 x 128

        x_e = Conv2DTranspose(base_filters * 2, 4, strides=(2, 2), padding='same', activation=keras_backend.relu)(x_e)  # 64 x 64 x 64
        x_e = Conv2DTranspose(base_filters * 1, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 64 x 64 x 64
        #x_e = Conv2DTranspose(base_filters, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 64 x 64 x 64

        x_e = Conv2DTranspose(4, 2, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 128 x 128 x 32
        x_e = Conv2DTranspose(2, 2, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 128 x 128 x 32

        #x_e = Conv2DTranspose(base_filters * 1, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 256 x 256 x 16
        #x_e = Conv2DTranspose(base_filters * 1, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)  # 256 x 256 x 16

        x_e = Conv2DTranspose(1, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)
        x_e = Conv2DTranspose(1, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)
        #x_e = Conv2DTranspose(1, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_e)   # 256x256x3

        x_e = Flatten()(x_e)

        # Fully connected layers to reshape into exact right amount of points

        x_e = Dense(N*3, activation=keras_backend.relu)(x_e)

        x_e = Reshape((N, 3))(x_e)

        model = Model(inputs, x_e)
        return model


if __name__=='__main__':
    model = get_keras_model(8, 3, (5, 256, 256, 3))

    print(model.summary())
