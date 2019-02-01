# IMPORT
import argparse #for command line arguments
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
matplotlib.use('Agg')


#for debugging, allows for reproducible (deterministic) results
#np.random.seed(0)


#to save our model periodically as checkpoints for loading later
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K
from models import nvidia_model, nvidia_model_tuned,shift_model,nvidia_model_basic,\
resnet50_pre_trained_model,nvidia_model2, nvidia_model_basic2, nvidia_model_tuned2,shift_model2

from functions import predict, predict_shift, split_dataset_random,angle_loss,split_seq_dataset_groups,generator_seq_dataset_groups
parser = argparse.ArgumentParser()
from sklearn.utils import shuffle
import models as models
import functions as fn
import numpy as np

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

#PARSER
def initparse():

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--loss', default='l2')
    parser.add_argument('--direction', default='center')
    parser.add_argument('--train_dir', default='Ch2/')
    parser.add_argument('--val_random', action='store_true')
    parser.add_argument('--drop_low_angles', action='store_true')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--steer_threshold', default=0.03, type=float)
    parser.add_argument('--steer_correction', default=0.1, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--num_samples', default=-1, type=int)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--pretrained', default='None')
    parser.add_argument('--save_model', default='None')
    parser.add_argument('--output', default='angle')
    parser.add_argument('--window_len', default=4, type=int)
    parser.add_argument('--mode', default='concat')
    parser.add_argument('--lookahead', action='store_true')
    parser.add_argument('--folds', default=4, type=int)
    parser.add_argument('--root_dir', default='/Users/project ')

def loadargs_model(args):
    print("LOAD LOCAL ARGS")
    args.val_random = True
    #args.num_samples = -1
    args.output = 'angle'
    args.l2reg = 0.0
    args.root_dir = ""
    args.direction = 'center'
    args.steer_threshold = 0.03
    args.drop_low_angles = False
    args.steer_correction = 0.1
    args.lr = 1e-3
    args.loss = '12'
    args.train_dir = 'Ch2/'
    args.augmentation = False
    args.batch_size = 16
    args.num_workers = 4
    args.num_epochs = 14
    args.use_gpu = True
    args.pretrained = 'out/mymodel'
    args.save_model = 'out/mymodel'
    args.save_dir = 'out/'
    args.lookahead = True
    args.folds = 4
    args.mode = 'concat'
    args.window_len=4
    args.greyscale=True
    args.save_model_steps = 'None'
    args.num_samples = -1
    args.num_samples_va=1000
    args.num_frames=10
    args.lookahead_window=30
    args.bad_angles=True
    args.video_angles=0
    args.vide_pred = 1
    #print(args)

#TRIAN FNs
def train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen):
    """
        Train the model
    """
    # calculate the difference between expected steering angle and actual steering angle
    # square the difference
    # add up all those differences for as many data points as we have
    # divide by the number of them
    # that value is our mean squared error! this is what we want to minimize via
    # gradient descent
    adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer=adam, metrics=[angle_loss])

    # callbacks = [EarlyStopping(monitor='val_loss',patience=2,verbose=0)]
    # Saves the model after every epoch.
    # quantity to monitor, verbosity i.e logging mode (0 or 1),
    # if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    # mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc,
    # this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    if args.save_model_steps != 'None':
        filepath = args.save_model + "-{val_angle_loss:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='angle_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
    else:
        callbacks_list = []

    # Fits the model on data generated batch-by-batch by a Python generator.

    # The generator is run in parallel to the model, for efficiency.
    # For instance, this allows you to do real-time data augmentation on images on CPU in
    # parallel to training your model on GPU.
    # so we reshape our data into their appropriate batches and train our model simulatenously
    model.summary()
    history_object = model.fit_generator(train_gen,
                                         steps_per_epoch=(len(train_imgs) / args.batch_size),
                                         validation_data=val_gen,
                                         validation_steps=(len(val_imgs) / args.batch_size),
                                         epochs=args.num_epochs,
                                         verbose=1,
                                         callbacks=callbacks_list)

    print("Finish history_object")
    if args.save_model != 'None':
        #model_json = model.to_json()
        #with open("modelold.json", "w") as json_file:
        #    json_file.write(model_json)
        model.save_weights("{}.h5".format(args.save_model))

    print("Init plot")

    ### plot the training and validation loss for each epoch
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(args.save_model +'loss.png')
    print("Validation loss for  "+ args.save_model )
    print(history_object.history['val_loss'])
    print("Loss for  " + args.save_model)
    print(history_object.history['loss'])
    print("Finish")

def train_model_custom(model, args, train_imgs, val_imgs, train_gen, val_gen):
    """
        Train the model
    """
    # calculate the difference between expected steering angle and actual steering angle
    # square the difference
    # add up all those differences for as many data points as we have
    # divide by the number of them
    # that value is our mean squared error! this is what we want to minimize via
    # gradient descent
    model.compile(loss="mean_squared_error", optimizer='adadelta', metrics=[fn.rmse])

    # Check a batch to see the base of the model
    # print(util.std_evaluate(model, util.generator_seq_dataset_chunks(validation_labels, validation_index_center, image_base_path_validation, 32, number_of_frames=num_frames), validation_index_center.shape[0]//32))
    history = fn.LossHistory()

    if args.save_model_steps != 'None':
        filepath = args.save_model + "-{val_angle_loss:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='angle_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [history,checkpoint]
    else:
        callbacks_list = [history]

    model.summary()


    # Fits the model on data generated batch-by-batch by a Python generator.

    # The generator is run in parallel to the model, for efficiency.
    # For instance, this allows you to do real-time data augmentation on images on CPU in
    # parallel to training your model on GPU.
    # so we reshape our data into their appropriate batches and train our model simulatenously

    history_object = model.fit_generator(train_gen,
                                         steps_per_epoch=(len(train_imgs) / args.batch_size),
                                         validation_data=val_gen,
                                         validation_steps=(len(val_imgs) / args.batch_size),
                                         epochs=args.num_epochs,
                                         verbose=1,
                                         callbacks=callbacks_list)

    print("Finish history_object")
    if args.save_model != 'None':
        #model_json = model.to_json()
        #with open("modelold.json", "w") as json_file:
        #    json_file.write(model_json)
        model.save_weights("{}.h5".format(args.save_model))

    print("Init plot")

    ### plot the training and validation loss for each epoch
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(args.save_model +'loss.png')
    print("Validation loss for  "+ args.save_model )
    print(history_object.history['val_loss'])
    print("Loss for  " + args.save_model)
    print(history_object.history['loss'])
    print("Finish")


# TEST models
def LSTM_test(args, train_imgs, val_imgs, train_gen, val_gen):
    # build_model_nvidia_wt_LSTM_TL2

    # chaffeur

    args.pretrained = 'out/mymodelLSTM_multiple_input'
    args.save_model = 'out/mymodelLSTM_multiple_input'

    model = models.LSTM_multiple_input(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        print(fn.std_evaluate(model,val_gen,(len(val_imgs) / args.batch_size)))
        #predict(model, args.pretrained, args)

def LSTM_test_adam(args, train_imgs, val_imgs, train_gen, val_gen):
    # build_model_nvidia_wt_LSTM_TL2

    # conv3d
    print("conv3d")
    args.pretrained = 'out/mymodelconv3d'
    args.save_model = 'out/mymodelconv3d'

    model = models.conv3d_model(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))
        #predict(model, args.pretrained, args)


    # lstm2d

    args.pretrained = 'out/mymodellstm2d'
    args.save_model = 'out/mymodellstm2d'

    model = models.conv2d_res_lstm_model(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        print(fn.std_evaluate(model,val_gen,(len(val_imgs) / args.batch_size)))
        #predict(model, args.pretrained, args)

def LSTM_img_sharing(args, train_imgs, val_imgs, train_gen, val_gen):



    # LSTM_img_sharing
    print("LSTM_img_sharing")
    args.pretrained = 'out/mymodelLSTM_img_sharing'
    args.save_model = 'out/mymodelLSTM_img_sharing'

    model = models.LSTM_img_sharing(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))
        #predict(model, args.pretrained, args)

    # lstm2

    args.pretrained = 'out/LSTM_multiple_input'
    args.save_model = 'out/LSTM_multiple_input'

    model = models.LSTM_multiple_input(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))
        # predict(model, args.pretrained, args)

def LSTM_img_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen):
    # CONV2D_img_sharing_ahead
    args.lookahead_window = 30
    args.num_frames = 10
    print("LSTM_img_sharing_ahead")
    args.pretrained = 'out/mymodelLSTM_img_sharing_ahead1_f10'
    args.save_model = 'out/mymodelLSTM_img_sharing_ahead1f_10'
    print(args)
    model = models.LSTM_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

    args.num_frames = 1
    print("LSTM_img_sharing_ahead")
    args.pretrained = 'out/mymodelLSTM_img_sharing_ahead1_f_1'
    args.save_model = 'out/mymodelLSTM_img_sharing_ahead1f_1'
    print(args)
    model = models.LSTM_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

    args.num_frames = 5
    print("LSTM_img_sharing_ahead")
    args.pretrained = 'out/mymodelLSTM_img_sharing_ahead1_f_5'
    args.save_model = 'out/mymodelLSTM_img_sharing_ahead1f_5'
    print(args)
    model = models.LSTM_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

    args.num_frames = 15
    print("LSTM_img_sharing_ahead")
    args.pretrained = 'out/mymodelLSTM_img_sharing_ahead1_f_15'
    args.save_model = 'out/mymodelLSTM_img_sharing_ahead1f_15'
    print(args)
    model = models.LSTM_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

    args.num_frames = 30
    print("LSTM_img_sharing_ahead")
    args.pretrained = 'out/mymodelLSTM_img_sharing_ahead1_f_30'
    args.save_model = 'out/mymodelLSTM_img_sharing_ahead1f_30'
    print(args)
    model = models.LSTM_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))


    args.num_frames = 60
    print("LSTM_img_sharing_ahead")
    args.pretrained = 'out/mymodelLSTM_img_sharing_ahead1_f_60'
    args.save_model = 'out/mymodelLSTM_img_sharing_ahead1f_60'
    print(args)
    model = models.LSTM_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))


def Nvidia_img_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen):


    print("nvidia_img_sharing_ahead")
    args.pretrained = 'out/mymodelnvidia_img_sharing_ahead'
    args.save_model = 'out/mymodelnvidia_img_sharing_ahead'

    model = models.nvidia_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))
        #predict(model, args.pretrained, args)

        # LSTM_img_sharing_ahead
    print("nvidia_img_sharing_ahead2")
    args.pretrained = 'out/mymodelnvidia_img_sharing_ahead2'
    args.save_model = 'out/mymodelnvidia_img_sharing_ahead2'

    model = models.nvidia_img_sharing_ahead2(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))
        # predict(model, args.pretrained, args)

def Nvidia_test(args, train_imgs, val_imgs, train_gen, val_gen):


    """""""""
    args.pretrained = 'out/mymodelnvidia_model_basic'
    args.save_model = 'out/mymodelnvidia_model_basic'
    model = nvidia_model_basic()
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        predict(model, args.pretrained, args)

    # nvidia_model_basic2

    args.pretrained = 'out/mymodelnvidia_model_basic2'
    args.save_model = 'out/mymodelnvidia_model_basic2'
    model = nvidia_model_basic2()
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        predict(model, args.pretrained, args)
        # nvidia_model_basic3
        
    # nvidia_model_basic4

    args.pretrained = 'out/mymodelnvidia_model_basic4'
    args.save_model = 'out/mymodelnvidia_model_basic4'
    model = models.nvidia_model_basic4()
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        predict(model, args.pretrained, args)

    # nvidia_tuned

    args.pretrained = 'out/mymodelnvidia_tuned'
    args.save_model = 'out/mymodelnvidia_tuned'
    num_outputs = 1 if args.output == 'angle' else 3
    model = nvidia_model_tuned(num_outputs, args.l2reg)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        predict(model, args.pretrained, args)

    # nvidia_tuned2

    args.pretrained = 'out/mymodelnvidia_tuned2'
    args.save_model = 'out/mymodelnvidia_tuned2'
    num_outputs = 1 if args.output == 'angle' else 3
    model = nvidia_model_tuned2(num_outputs, args.l2reg)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        predict(model, args.pretrained, args)
        
    """""""""
    print(args)
    args.pretrained = '1img/mymodelnvidia_model_basic'+ str(args.lookahead_window) +str(args.num_frames)
    args.save_model = '1img/mymodelnvidia_model_basic'+ str(args.lookahead_window) +str(args.num_frames)
    model = models.nvidia_model_basic_cropped()
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_just_ahead_nvidia(model, args.pretrained, args)




def Transfer_Learning_test(args, train_imgs, val_imgs, train_gen, val_gen):
    # resnet50_pre_trained_model
    """
    args.pretrained = 'out/mymodelbuild_model_resnet50_pre_trained'
    args.save_model = 'out/mymodelbuild_model_resnet50_pre_trained'
    model = resnet50_pre_trained_model()
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        predict(model, args.pretrained, args)
    """

    # build_model_resnet50_fully
    """
    args.pretrained = 'out/mymodelbuild_model_resnet50_fully'
    args.save_model = 'out/mymodelbuild_model_resnet50_fully'
    model = build_model_resnet50_fully()
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        predict(model, args.pretrained, args)
     """

def Data_sharing_test(args, train_imgs, val_imgs, train_gen, val_gen):
    args.pretrained = 'out/mymodelshift'
    args.save_model = 'out/mymodelshift'
    num_outputs = 1 if args.output == 'angle' else 3
    num_inputs = 2 if args.mode == 'concat' else 1
    if args.greyscale == False:
        num_inputs = num_inputs * 3
    model = shift_model(num_outputs, args.l2reg, num_inputs)
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    print(train_imgs.shape)
    if args.pretrained != 'None':
        predict_shift(model, args.pretrained, args)

    # Shift dataset 2images gray

    args.pretrained = 'out/mymodelshift2'
    args.save_model = 'out/mymodelshift2'
    num_outputs = 1 if args.output == 'angle' else 3
    num_inputs = 2 if args.mode == 'concat' else 1
    if args.greyscale == False:
        num_inputs = num_inputs * 3
    model = shift_model2(num_outputs, args.l2reg, num_inputs)
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    print(train_imgs.shape)
    if args.pretrained != 'None':
        predict_shift(model, args.pretrained, args)

    # Shift dataset images dif gray

    args.mode = 'diff'
    train_gen = generator_seq_dataset_groups(train_imgs, args)
    val_gen = generator_seq_dataset_groups(val_imgs, args)

    args.pretrained = 'out/mymodelshift_diff'
    args.save_model = 'out/mymodelshift_diff'
    num_outputs = 1 if args.output == 'angle' else 3
    num_inputs = 2 if args.mode == 'concat' else 1
    if args.greyscale == False:
        num_inputs = num_inputs * 3
    model = shift_model(num_outputs, args.l2reg, num_inputs)
    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    print(train_imgs.shape)
    if args.pretrained != 'None':
        predict_shift(model, args.pretrained, args)


def CONv3D_img_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen):

    # CONV3D_img_sharing_ahead
    print("Conv3D_img_sharing_ahead")
    args.pretrained = 'out/mymodelConv3D_img_sharing_ahead1'
    args.save_model = 'out/mymodelConv3D_img_sharing_ahead1'

    model = models.Con3d_model(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':

        fn.predict_temporal(model, args.pretrained, args)
        #print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

        # CONV3D_img_sharing_ahead
    print("Conv3D_img_sharing_ahead2")
    args.pretrained = 'out/mymodelConv3D_img_sharing_ahead22'
    args.save_model = 'out/mymodelConv3D_img_sharing_ahead22'


    model = models.Con3d_model2(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

        # CONV3D_img_sharing_ahead
    print("Conv3D_img_sharing_ahead3")
    args.pretrained = 'out/mymodelConv3D_img_sharing_ahead33'
    args.save_model = 'out/mymodelConv3D_img_sharing_ahead33'

    model = models.Con3d_model3(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

        # CONV3D_img_sharing_ahead
    print("Conv3D_img_sharing_ahead4")
    args.pretrained = 'out/mymodelConv3D_img_sharing_ahead44'
    args.save_model = 'out/mymodelConv3D_img_sharing_ahead44'

    model = models.Con3d_model4(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

def CONv2D_img_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen):
    # CONV2D_img_sharing_ahead
    print("Conv2D_img_sharing_ahead")
    args.pretrained = 'out/mymodelConv2D_img_sharing_ahead1'
    args.save_model = 'out/mymodelConv2D_img_sharing_ahead1'
    print(args)
    model = models.nvidia_img_sharing_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

    print("Conv2D_img_sharing_ahead2")
    args.pretrained = 'out/mymodelConv2D_img_sharing_ahead2'
    args.save_model = 'out/mymodelConv2D_img_sharing_ahead2'
    print(args)
    model = models.nvidia_img_sharing_ahead2(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_temporal(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

def CONv3D_LSTMimg_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen):

    # CONV3D_img_sharing_ahead
    print("Conv3DLSTM_img_sharing_ahead")
    args.pretrained = 'out/mymodelConv3DLSTM_img_sharing_ahead1'
    args.save_model = 'out/mymodelConv3DLSTM_img_sharing_ahead1'

    model = models.Con3d_model_LSTM(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':

        fn.predict_temporal_timedistrib(model, args.pretrained, args)
        #print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))
def CONv3D_img_sharing_Just_ahead(args, train_imgs, val_imgs, train_gen, val_gen):

    print(args)
    # LSTM_img_sharing_Just_ahead
    print("LSTM_img_sharing_Just_ahead")
    args.pretrained = 'out/mymodelLSTM_img_sharing_Just_ahead'
    args.save_model = 'out/mymodelLSTM_img_sharing_Just_ahead'

    model = models.LSTM_img_sharing(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_just_ahead(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

    # CONv3D_img_sharing_Just_ahead
    print("CONv3D_img_sharing_Just_ahead")
    args.pretrained = 'out/mymodelCONv3D_img_sharing_Just_ahead'
    args.save_model = 'out/mymodelCONv3D_img_sharing_Just_ahead'

    model = models.Con3d_model_just_ahead(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':

        fn.predict_just_ahead(model, args.pretrained, args)
        #print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

def CONv2D_img_diff(args, train_imgs, val_imgs, train_gen, val_gen):

    print(args)
    # CONv2D_img_diff
    print("CONv2D_img_diff")
    args.pretrained = 'out/mymodelCONv2D_img_diff22' + str(args.lookahead_window) +str(args.num_frames)
    args.save_model = 'out/mymodelCONv2D_img_diff22' + str(args.lookahead_window) +str(args.num_frames)

    model = models.nvidia_model_diff(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':
        fn.predict_diff(model, args.pretrained, args)
        # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

def CONv2D_img_concat(args, train_imgs, val_imgs, train_gen, val_gen):

    print(args)
    # CONv2D_img_concat
    print("CONv2D_img_concat")
    args.pretrained = 'out/mymodelCONv2D_img_concat22' + str(args.lookahead_window) +str(args.num_frames)
    args.save_model = 'out/mymodelCONv2D_img_concat22'+ str(args.lookahead_window) +str(args.num_frames)

    model = models.nvidia_model_concat(args)

    train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

    if args.pretrained != 'None':

        fn.predict_diff(model, args.pretrained, args)
        #print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))


def main(args):
    loadargs_model(args)
    # for debug reduces number of samples
    args.num_samples = -1
    args.num_samples_va=50
    args.num_frames=5
    args.lookahead_window=5
    args.bad_angles=False
    args.angles_degree=False
    args.use_more_big_angles=False
    args.video_angles=1
    args.vide_pred = 0
    args.batch_size = 32
    args.num_epochs = 14

    test_nvidia=0
    test_nvidia_future_img=0
    made_video=0
    test_predictions=1
    test_conv3d=0
    test_con2d_ahead=0
    test_lstm=0
    test_lstm_2=0
    test_conv3d_lstm =0
    test_conv3d_just_ahead=0
    test_conv2d_diff=0
    test_lstm_just_ahead=0

    print(args)
    if test_nvidia==1:
        # Basic dataset
        train_imgs, val_imgs = split_dataset_random(args)
        if args.num_samples > 0:
            train_imgs = train_imgs[:args.num_samples]
            val_imgs = val_imgs[:args.num_samples_va]

        train_gen = fn.generator_dataset_random(train_imgs, args)
        val_gen = fn.generator_dataset_random(val_imgs, args)

        Nvidia_test(args, train_imgs, val_imgs, train_gen, val_gen)

    if test_nvidia_future_img==1:
        # Basic dataset
        for i in range(0,100,5):
            args.lookahead_window=i
            train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
            if args.num_samples > 0:
                train_imgs = train_imgs[:args.num_samples]
                val_imgs = val_imgs[:args.num_samples_va]

            train_gen = fn.generator_dataset_just_ahead(train_imgs[:, 1], train_imgs[:, 0], args)
            val_gen = fn.generator_dataset_just_ahead(train_imgs[:, 1], train_imgs[:, 0], args)

            Nvidia_test(args, train_imgs, val_imgs, train_gen, val_gen)


    if test_con2d_ahead==1:
        # CONV2d
        args.num_frames = 20
        train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
        if args.num_samples > 0:
            train_imgs = train_imgs[:args.num_samples]
            val_imgs = val_imgs[:args.num_samples_va]

        train_gen = fn.generator_seq_dataset_chunks_ahead(train_imgs[:, 1], train_imgs[:, 0], args)
        val_gen = fn.generator_seq_dataset_chunks_ahead(val_imgs[:, 1], val_imgs[:, 0], args)

        CONv2D_img_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen)

    if test_lstm == 1:
        # LSTM
        args.num_frames = 10
        train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
        if args.num_samples > 0:
            train_imgs = train_imgs[:args.num_samples]
            val_imgs = val_imgs[:args.num_samples_va]

        train_gen = fn.generator_seq_dataset_chunks_ahead(train_imgs[:, 1], train_imgs[:, 0], args)
        val_gen = fn.generator_seq_dataset_chunks_ahead(val_imgs[:, 1], val_imgs[:, 0], args)

        LSTM_img_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen)

    if test_lstm_2 == 1:
        # LSTM
        args.num_frames = 8
        train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
        if args.num_samples > 0:
            train_imgs = train_imgs[:args.num_samples]
            val_imgs = val_imgs[:args.num_samples_va]

        train_gen = fn.generator_seq_dataset_chunks_ahead_v2(train_imgs[:, 1], train_imgs[:, 0], args)
        val_gen = fn.generator_seq_dataset_chunks_ahead_v2(val_imgs[:, 1], val_imgs[:, 0], args)

        args.lookahead_window = 30
        args.num_frames = 8
        print("LSTM_img_sharing_ahead")
        args.pretrained = 'out/mymodelLSTM_img_sharing_ahead30_f8'
        args.save_model = 'out/mymodelLSTM_img_sharing_ahead30f_8'
        print(args)
        model = models.LSTM_img_sharing_ahead(args)

        train_model_adam(model, args, train_imgs, val_imgs, train_gen, val_gen)

        if args.pretrained != 'None':
            fn.predict_temporalv2(model, args.pretrained, args)
            # print(fn.std_evaluate(model, val_gen, (len(val_imgs) / args.batch_size)))

    """""
        # Shift dataset 2images gray
        train_imgs, val_imgs = split_seq_dataset_groups(args)
        args.mode = 'concat'
        train_gen = generator_seq_dataset_groups(train_imgs, args)
        val_gen = generator_seq_dataset_groups(val_imgs, args)
    
    
        #LSTM
        args.num_frames = 20
        train_imgs, val_imgs = fn.split_seq_dataset_chunks(args)
        train_gen = fn.generator_seq_dataset_chunks(train_imgs[:, 1], train_imgs[:, 0], args, scale=1,
                                                          random_flip=False)
        val_gen = fn.generator_seq_dataset_chunks(val_imgs[:, 1], val_imgs[:, 0], args, scale=1, random_flip=False)
    
        LSTM_test(args, train_imgs, val_imgs, train_gen, val_gen)
        
           args.num_frames = 5
        train_imgs, val_imgs = fn.split_dataset_random(args)
    
        train_gen = fn.generator_dataset_random(train_imgs,args)
        val_gen = fn.generator_dataset_random(val_imgs,args)
    
        Nvidia_test(args, train_imgs, val_imgs, train_gen, val_gen)
    
    """""
    if test_conv3d==1:
        # CONV3d
        args.num_frames = 5
        train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
        if args.num_samples > 0:
            train_imgs = train_imgs[:args.num_samples]
            val_imgs = val_imgs[:args.num_samples_va]

        train_gen = fn.generator_seq_dataset_chunks_ahead(train_imgs[:, 1], train_imgs[:, 0],args)
        val_gen = fn.generator_seq_dataset_chunks_ahead(val_imgs[:, 1], val_imgs[:, 0], args)

        CONv3D_img_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen)

    if test_conv3d_lstm == 1:
        # CONV3d_LSTM
        args.num_frames = 10
        args.lookahead_window = 15
        train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
        if args.num_samples > 0:
            train_imgs = train_imgs[:args.num_samples]
            val_imgs = val_imgs[:args.num_samples_va]

        train_gen = fn.generator_seq_dataset_chunks_ahead_time(train_imgs[:, 1], train_imgs[:, 0], args)
        val_gen = fn.generator_seq_dataset_chunks_ahead_time(val_imgs[:, 1], val_imgs[:, 0], args)

        CONv3D_LSTMimg_sharing_ahead(args, train_imgs, val_imgs, train_gen, val_gen)

    if test_conv3d_just_ahead==1:
        # CONV3d
        args.num_frames = 25
        train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
        if args.num_samples > 0:
            train_imgs = train_imgs[:args.num_samples]
            val_imgs = val_imgs[:args.num_samples_va]

        train_gen = fn.generator_seq_dataset_just_ahead(train_imgs[:, 1], train_imgs[:, 0],args)
        val_gen = fn.generator_seq_dataset_just_ahead(val_imgs[:, 1], val_imgs[:, 0], args)

        CONv3D_img_sharing_Just_ahead(args, train_imgs, val_imgs, train_gen, val_gen)


    if test_conv2d_diff==1:

        args.lookahead_window = 15
        for i in range(2,22,4):

            args.num_frames = i
            train_imgs, val_imgs = fn.split_seq_dataset_simple(args)
            if args.num_samples > 0:
                train_imgs = train_imgs[:args.num_samples]
                val_imgs = val_imgs[:args.num_samples_va]



            args.mode = 'concat'

            train_gen = fn.generator_seq_dataset_diff_ahead(train_imgs[:, 1], train_imgs[:, 0], args)
            val_gen = fn.generator_seq_dataset_diff_ahead(val_imgs[:, 1], val_imgs[:, 0], args)

            CONv2D_img_concat(args, train_imgs, val_imgs, train_gen, val_gen)

            args.mode = 'diff'

            train_gen = fn.generator_seq_dataset_diff_ahead(train_imgs[:, 1], train_imgs[:, 0], args)
            val_gen = fn.generator_seq_dataset_diff_ahead(val_imgs[:, 1], val_imgs[:, 0], args)

            CONv2D_img_diff(args, train_imgs, val_imgs, train_gen, val_gen)


    if made_video==1:
        args.pretrained = "nvidia_model_basic_all_angles"
        model = models.nvidia_model_basic()
        fn.create_video_pre_process(args,model)

    if test_predictions==1:
        args.bad_angles = True
        args.pretrained="mymodelnvidia_tuned"
        model=models.nvidia_model_tuned(1,args.l2reg)
        predict(model, args.pretrained, args)
        #

        """
        args.bad_angles = False
        args.pretrained="out/mymodelnvidia_model_basic2"
        model=models.nvidia_model_basic3()
        predict(model, args.pretrained, args)
        
        args.pretrained="nvidia_model_basic_good_angles"
        model=models.nvidia_model_basic()
        predict(model, args.pretrained, args)
        
        args.pretrained = "mymodelConv3D_img_sharing_ahead"
        args.num_frames = 10
        args.lookahead_window = 15
        model=models.Con3d_model(args)
        fn.predict_temporal(model, args.pretrained, args)
        
                args.pretrained = "mymodelCONv2D_img_concat22"
        args.mode = 'concat'
        args.num_frames = 2
        model = models.nvidia_model_concat(args)
        rmse = []
        index=[]
        for i in range(0,100,5):
            args.lookahead_window = i

            rmsenum= fn.predict_diff(model, args.pretrained, args)

            rmse.append(rmsenum)
            index.append(i)
        
        rmse=np.array(rmse)
        index=np.array(index)
        plt.figure(figsize=(16, 9))
        plt.plot(index,rmse, label='RSME vs dt')

        plt.legend()
        plt.savefig('diferent_dt.png')

           """





if __name__ == '__main__':
  initparse()
  args = parser.parse_args()
  print("START MAIN WITH ARGS")
  print(args)
  with K.get_session():
      main(args)
