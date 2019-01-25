#IMPORT
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from tensorflow.python.keras.models import Sequential, Model
#what types of layers do we want our model to have?
from tensorflow.python.keras.layers import LSTM,Lambda, Dense, Dropout, Activation, Input,Flatten, Conv2D, MaxPooling2D, \
    Cropping2D, Dropout,TimeDistributed, Conv3D, MaxPooling3D,Cropping3D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf
#from tensorflow.python.keras.layers.wrappers import TimeDistributed
#Nvidia variations models
def nvidia_model(l2reg):
    # Keras NVIDIA type model
    model = Sequential()
    #model.add(Lambda(lambda x: x / 255.0 - 0, input_shape=(480, 640, 3)))
    #model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(480, 640, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros', input_shape=(480, 640, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(1, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))

    return model
def nvidia_model2(l2reg):
    # Keras NVIDIA type model
    model = Sequential()
    #model.add(Lambda(lambda x: x / 255.0 - 0, input_shape=(480, 640, 3)))
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(480, 640, 3)))
    model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(1, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))

    return model

def nvidia_model_tuned(num_outputs, l2reg):
    # Keras NVIDIA type model
    model = Sequential()
    #model.add(Lambda(lambda x: x / 255.0 - 0, input_shape=(480, 640, 3)))
    model.add(Cropping2D(cropping=((240, 0), (0, 0)), input_shape=(480, 640, 3)))  # trim image to only see section with road

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(num_outputs, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))

    return model
def nvidia_model_tuned2(num_outputs, l2reg):
    # Keras NVIDIA type model
    model = Sequential()
    #model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(480, 640, 3)))
    #model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros', input_shape=(480, 640, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(num_outputs, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))

    return model

def nvidia_model_basic():
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(480, 640, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(480, 640, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model
def nvidia_model_basic_cropped():
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(240, 640, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(240, 640, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model

def nvidia_model_diff(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(240*args.num_frames, 640, 3)))

    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(240*args.num_frames, 640, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(240*args.num_frames, 640, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2),activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model

def nvidia_model_concat(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(240*args.num_frames, 640, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(240*args.num_frames, 640, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2),activation='relu'))
    model.add(Conv2D(64, (3, 3),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model
def nvidia_model_basic2():
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(480, 640, 3)))
    #model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(480, 640, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model
def nvidia_model_basic3():
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(480, 640, 3)))
    #model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(480, 640, 3)))
    model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model
def nvidia_model_basic4():
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(480, 640, 3)))
    #model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road
   # model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(480, 640, 3)))
    model.add(Cropping2D(cropping=((240, 0), (0, 0)), input_shape=(480, 640, 3)))  # trim image to only see section with road
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model

def Con3d_model_just_ahead(args):
    """

    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255, input_shape=(args.num_frames,240, 640, 1)))

    model.add(Conv3D(24, (3, 3,3), strides=(2, 2,2), padding = 'same',activation='relu', input_shape=(args.num_frames,240, 640,1)))
    model.add(Conv3D(36, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(48, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    print("test")

    return model


def Con3d_model(args):
    """

    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(args.num_frames*2,480, 640, 3)))
    model.add(Conv3D(24, (3, 3,3), strides=(2, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(36, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(48, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    print("test")

    return model
def Con3d_model2(args):
    """

    """
    model = Sequential()
    model.add(Conv3D(24, (3, 3,3), strides=(2, 2,2), padding = 'same',activation='relu', input_shape=(args.num_frames*2,480, 640, 3)))
    model.add(Conv3D(36, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(48, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    print("test")

    return model
def Con3d_model3(args):
    """

    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(args.num_frames*2,480, 640, 3)))
    model.add(Cropping3D(cropping=((0,0), (240, 0), (0, 0))))
    model.add(Conv3D(24, (3, 3,3), strides=(2, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(36, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(48, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    print("test")

    return model
def Con3d_model4(args):
    """

    """
    model = Sequential()
    model.add(Cropping3D(cropping=((0,0), (240, 0), (0, 0)), input_shape=(args.num_frames*2,480, 640, 3)))
    model.add(Conv3D(24, (3, 3,3), strides=(2, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(36, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    model.add(Conv3D(48, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    print("test")

    return model



def Con3d_model2_modif(l2reg,args,num_frames):
    # Keras NVIDIA type model
    model = Sequential()
    #model.add(Lambda(lambda x: x / 255.0 - 0, input_shape=(480, 640, 3)))
    #model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(480, 640, 3)))
    model.add(Cropping3D(cropping=((0, 0),(240, 0), (0, 0))))  # trim image to only see section with road

    model.add(Conv3D(24, (5,5, 5), strides=(2, 2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros', input_shape=(num_frames, 480, 640, 3)))
    model.add(Conv3D(36, (3,3, 3), strides=(2, 2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv3D(48, (3,3, 3), strides=(2, 2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv3D(64, (3,3, 3),activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Conv3D(64, (3,3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dense(1, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))

    return model


def nvidia_img_sharing_ahead(args):
    """

    """
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(480, 640, 3)))
    #model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(2*args.num_frames,480, 640, 3)))

    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(2 * args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Flatten()))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model
def nvidia_img_sharing_ahead2(args):
    """

    """
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(480, 640, 3)))
    #model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(2*args.num_frames,480, 640, 3)))
    model.add(Cropping3D(cropping=((0,0),(240, 0), (0, 0))))  # trim image to only see section with road

    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(2 * args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Flatten()))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model

#Tranfer learning models
def resnet50_pre_trained_model():
    """
    RESNET50 pre-trained Transfer Learning
    Get the model. Remove final layers and attach my own fully
    connected layers.
    Flatten : 2048
    Fully connected: 1024
    Fully connected: 512
    Fully connected: 256
    Fully connected: 128
    Fully connected: 64
    Fully connected: 1
    """
    # IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
    #    image_input = Input(shape=(224, 224, 3))
    image_input = Input(shape=(480, 640, 3))
    base_model = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet')
    base_model.summary()
    last_layer = base_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation='softmax', name='output_layer')(x)
    model = Model(inputs=image_input, outputs=out)
    #    model = Model(input=image_input,output= out)
    model.summary()
    for layer in model.layers[:-7]:
        layer.trainable = False

    return model

#Image sharing models
def shift_model(num_outputs, l2reg, num_inputs):
    # Keras NVIDIA type model
    model = Sequential()

    #model.add(Lambda(lambda x: x / 255.0, input_shape=(480, 640, num_inputs)))
   # model.add(Cropping2D(cropping=((240, 0), (150, 150))))  # trim image to only see section with road
    model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization(num_outputs))
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(num_outputs, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))

    return model
def shift_model2(num_outputs, l2reg, num_inputs):
    # Keras NVIDIA type model
    model = Sequential()

    #model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(480, 640,num_inputs )))
    #model.add(Lambda(lambda x: x / 255.0, input_shape=(480, 640, num_inputs)))
   # model.add(Cropping2D(cropping=((240, 0), (150, 150))))  # trim image to only see section with road
    #model.add(Cropping2D(cropping=((240, 0), (0, 0))))  # trim image to only see section with road

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization(num_outputs))
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(num_outputs, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(l2reg)))

    return model

def LSTM_simple_input(args):
    model = Sequential()

    #model.add(TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0), input_shape=(480, 640, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid',input_shape=(480, 640, 3))))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(BatchNormalization())
    model.add(LSTM(64,  return_sequences=True, implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(128,
        kernel_initializer='he_normal',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dropout(0.2))
    model.add(Dense(1,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(1e-3)))
    return model
def LSTM_multiple_input(args):
    model = Sequential()

    #model.add(TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0), input_shape=(args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64, implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(128,
        kernel_initializer='he_normal',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dropout(0.2))
    model.add(Dense(1,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(1e-3)))
    return model
def LSTM_multiple_input_v2(args):
    model = Sequential()

    #model.add(TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0), input_shape=(args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=True, implementation=2))
    #model.add(LSTM(64, return_sequences=True, implementation=2))
    #model.add(LSTM(64, implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(1,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(1e-3)))
    return model


def LSTM_img_sharing(args):

    model = Sequential()
    model.add(Lambda(lambda x: x/255, input_shape=(args.num_frames,240, 640, 1)))

    #model.add(TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0), input_shape=(args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(args.num_frames, 240, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64,  return_sequences=True, implementation=2))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64,  implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()


    return model
def LSTM_img_sharing_ahead(args):
    ##image sharing
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(2*args.num_frames,480, 640, 3)))
    #model.add(TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0), input_shape=(2*args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(2*args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64,  return_sequences=True, implementation=2))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64,  implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()


    return model
def LSTM_img_sharing_ahead_v2(args):
    ##image sharing
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(2 * args.num_frames, 480, 640, 3)))
    model.add(Cropping3D(cropping=((0, 0), (240, 0), (0, 0))))  # trim image to only see section with road

    #model.add(TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0), input_shape=(2*args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(2*args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(Dense(1))
    model.summary()


    return model


def Con3d_model_LSTM(args):
    """

    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(2,args.num_frames,240, 640, 3)))
    #model.add(Cropping3D(cropping=((0,0), (240, 0), (0, 0))))
    model.add(TimeDistributed(Conv3D(24, (3, 3,3), strides=(2, 2,2), padding = 'same',activation='relu')))
    model.add(TimeDistributed(Conv3D(36, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu')))
    model.add(TimeDistributed(Conv3D(48, (3, 3,3), strides=(1, 2,2), padding = 'same',activation='relu')))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    #model.add(Conv3D(64, (3, 3,3), activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64, implementation=2))
    #model.add(Dropout(0.2))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    print("test")

    return model

#conv3D
def conv2d_res_lstm_model(args):
    xin = Input(batch_shape=(args.batch_size, args.num_frames, 480, 640, 3))
    con2d1 = TimeDistributed(Conv2D(3, (5, 5), strides=(2, 2), activation='relu', use_bias=True, padding='SAME'))(xin)
    #print(con2d1._keras_shape, "con3d1")
    cs = BatchNormalization(axis=3)(con2d1)
    #print(cs._keras_shape, "bn1")

    cs = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(cs)
    #print(cs._keras_shape, "max")

    for i in range(4):
        c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', use_bias=True, padding='SAME'))(cs)
        bn1 = BatchNormalization(axis=3)(c)
        c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', use_bias=True, padding='SAME'))(bn1)
        bn = BatchNormalization(axis=3)(c)
        c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', use_bias=True, padding='SAME'))(c)
        bn = BatchNormalization(axis=3)(c)
        cs = tf.keras.layers.add([bn1, bn])
        #print(cs._keras_shape, "res_block{}".format(i))

    for i in range(4):
        c = TimeDistributed(Conv2D(64, (1, 1), strides=(2, 2), activation='relu', use_bias=True, padding='VALID'))(cs)
        cs = BatchNormalization(axis=3)(c)
        #print(cs._keras_shape, "shrink_block{}".format(i))

    flt = TimeDistributed(Flatten())(cs)
    #print(flt._keras_shape)
    lstm1 = LSTM(64, activation='tanh', return_sequences=True, implementation=2)(flt)
    #print(lstm1._keras_shape, "lstm1")
    lstm2 = LSTM(16, activation='tanh', return_sequences=True, implementation=2)(lstm1)
    #print(lstm2._keras_shape, "lstm2")
    lstm3 = LSTM(16, activation='tanh')(lstm2)
    dense1 = Dense(512, activation='relu', use_bias=True)(lstm3)
    #print(dense1._keras_shape, "dense1")
    dense2 = Dense(256, activation='relu', use_bias=True)(dense1)
    #print(dense2._keras_shape, "dense2")
    dense3 = Dense(64, activation='relu', use_bias=True)(dense2)
    #print(dense3._keras_shape, "dense3")
    angle = Dense(1)(dense3)
    #print(angle._keras_shape, "angle")
    model = Model(inputs=xin, outputs=angle)

    return model
def conv3d_model(args):
    input_layer = Input(batch_shape=(args.batch_size, args.num_frames, 480, 640, 3))

    c = Conv3D(24, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME')(input_layer)
    cs = BatchNormalization(axis=4)(c)
    #print(cs._keras_shape, "3d1")
    cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)
    cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)

    for i in range(4):
        c = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME')(cs)
        bn1 = BatchNormalization(axis=4)(c)
        c = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME')(bn1)
        bn = BatchNormalization(axis=4)(c)
        c = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME')(c)
        bn = BatchNormalization(axis=4)(c)
        cs = tf.keras.layers.add([bn1, bn])
        #print(cs._keras_shape, "res_block{}".format(i))

    cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)
    cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)

    # for i in range(1):
    #     c = Conv3D(24, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='VALID')(cs)
    #     cs = BatchNormalization(axis=3)(c)
    #     print(cs._keras_shape, "shrink_block{}".format(i))

    flt = Flatten()(cs)
    #dense1 = Dense(512, activation='relu', use_bias=True)(flt)
    dense1 = Dense(256, activation='relu', use_bias=True)(flt)
    #print(dense1._keras_shape, "dense1")
    dense2 = Dense(128, activation='relu', use_bias=True)(dense1)
    #print(dense2._keras_shape, "dense2")
    dense3 = Dense(64, activation='relu', use_bias=True)(dense2)
    #print(dense3._keras_shape, "dense3")
    angle = Dense(1, use_bias=True)(dense3)
    #print(angle._keras_shape, "angle")


    model = Model(inputs=input_layer, outputs=angle)

    return model

#Previus models
def Chaffeur_model():
    num_frames = 50
    model = Sequential()
    model.add(TimeDistributed(Conv2D(24, 5, 5,
                                     init="he_normal",
                                     activation='relu',
                                     subsample=(5, 4),
                                     border_mode='valid'), input_shape=(num_frames, 120, 320, 3)))
    model.add(TimeDistributed(Conv2D(32, 5, 5,
                                     init="he_normal",
                                     activation='relu',
                                     subsample=(3, 2),
                                     border_mode='valid')))
    model.add(TimeDistributed(Conv2D(48, 3, 3,
                                     init="he_normal",
                                     activation='relu',
                                     subsample=(1, 2),
                                     border_mode='valid')))
    model.add(TimeDistributed(Conv2D(64, 3, 3,
                                     init="he_normal",
                                     activation='relu',
                                     border_mode='valid')))
    model.add(TimeDistributed(Conv2D(128, 3, 3,
                                     init="he_normal",
                                     activation='relu',
                                     subsample=(1, 2),
                                     border_mode='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True, implementation=2))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True, implementation=2))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=256,
        init='he_normal',
        activation='relu',
        W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=1,
        init='he_normal',
        W_regularizer=l2(0.001)))
def Chauffeur_lstm_model(args):

    model = Sequential()
    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(5, 4),
                                     padding='valid'), input_shape=(args.num_frames, 480, 640, 3)))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(3, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(1, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64,  return_sequences=True, implementation=2))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64,  implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(256,
        kernel_initializer='he_normal',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dropout(0.2))
    model.add(Dense(1,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(1e-3)))
    model.summary()
    return model

