#IMPORT
import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import cv2
import csv
import matplotlib
import keras
import numpy as np
import os
import random
import keras.backend as K
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend('agg')
matplotlib.use('Agg')

print ("Import libs")
from sklearn.utils import shuffle
#to split out training and testing data
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from PIL import Image, ImageOps

radian_to_degree= 57.2958
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
img_shape = (640, 480)
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


# main functions
def load_images(args):
    images = []
    # From test images Ch2/
    with open('Ch2/interpolated.csv') as csvfile:
        if args.direction == 'center':
            _directions = ['center']
        elif args.direction == 'left':
            _directions = ['left']
        elif args.direction == 'left':
            _directions = ['right']
        else:
            _directions = ['center', 'left', 'right']

        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            d = row[5].split('/')[0]
            path = 'Ch2/' + d + '/' + row[5].split('/')[1]
            angle = float(row[6])
            torque = float(row[7])
            speed = float(row[8])

            if d in _directions:
                if d == 'center':
                    ix = 0
                elif d == 'left':
                    ix = 1
                else:
                    ix = 2
                item = (path, angle, speed, torque, ix, 0)
                images.append(item)
    return np.array(images)
def preprocess(samples, steer_threshold, drop_low_angles):
    samples = np.array(samples, dtype = object)
    print("Number of samples before dropping low steering angles: {}".format(samples.shape[0]))
    #index1 = np.where((samples[:, 1] < 0) == True)[0]
    #print(index1)
    index = np.where( (np.abs(samples[:,1]) < steer_threshold) == True)[0]

    if drop_low_angles == False:
        rows = [i for i in index if np.random.randint(10) < 9]
    else:
        rows = index
    samples = np.delete(samples, rows, 0)
    print("Removed %s rows with low steering"%(len(rows)))
    print("Number of samples after dropping low steering angles: {}".format(samples.shape[0]))

    return samples
def crop(image):
    """
    Crop the image
    """
    return image[240:480, :, :] # remove the sky and the car front
def preprocess_image(image):
    image=crop(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image=cv2.equalizeHist(image)

    #img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    #img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    #image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    #image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    # img = ((img - (255.0 / 2)) / 255.0)
    return image
def correct_steering(angle, ix, steer_correction):
    if ix == 1: #left
        return angle + steer_correction
    elif ix == 2: #right
        return angle - steer_correction
    else:
        return angle

def create_video(args,model=0):
    images = []
    images_count=0
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if args.video_angles == 1:
        out = cv2.VideoWriter('output_badangles.avi', fourcc, 5.0, (640, 480))
    else:
        out = cv2.VideoWriter('output_allangles.avi', fourcc, 15.0, (640, 480))
    with open('Ch2/interpolated.csv') as csvfile:
        if args.direction == 'center':
            _directions = ['center']
        elif args.direction == 'left':
            _directions = ['left']
        elif args.direction == 'left':
            _directions = ['right']
        else:
            _directions = ['center', 'left', 'right']

        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:

            d = row[5].split('/')[0]
            path = 'Ch2/' + d + '/' + row[5].split('/')[1]
            angle = float(row[6])
            torque = float(row[7])
            speed = float(row[8])

            if d in _directions:
                if d == 'center':
                    ix = 0
                elif d == 'left':
                    ix = 1
                else:
                    ix = 2
                #if images_count > 100:
                #    break
                images_count = images_count + 1

                #angle = correct_steering(angle, ix, args.steer_correction)
                #item = (path, angle, speed, torque, ix, 0)
                #img = pil_loader(path)
                #frame = np.asarray(img)
                #frame=np.load(path)
                if args.video_angles==1:

                    if angle*57.2958 <-30  or  angle*57.2958 >30  or angle==0  :
                        frame=cv2.imread(path)
                        if args.vide_pred==1:
                            angle_pred=model.predict(frame.reshape(-1, 480, 640, 3))[0][0]
                            cv2.putText(frame, str(angle_pred*57.2958), (300, 120), font, 1.0, (255, 0, 0), 3)
                        cv2.putText(frame, str(angle), (300,50), font, 1.0, (0, 0, 255), 3)
                        cv2.putText(frame, str(angle*57.2958), (300, 80), font, 1.0, (0, 0, 255), 3)
                        ##frame =cv2.imread(path)
                        out.write(frame)

                else:
                    frame = cv2.imread(path)
                    if args.vide_pred == 1:
                        angle_pred = model.predict(frame.reshape(-1, 480, 640, 3))[0][0]
                        cv2.putText(frame, str(angle_pred * 57.2958), (300, 120), font, 1.0, (255, 0, 0), 3)
                    cv2.putText(frame, str(angle), (300, 50), font, 1.0, (0, 0, 255), 3)
                    cv2.putText(frame, str(angle * 57.2958), (300, 80), font, 1.0, (0, 0, 255), 3)
                    ##frame =cv2.imread(path)
                    out.write(frame)

                ##cv2.imshow('frame', frame)

    print("Ended Video")
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
def create_video_pre_process(args,model=0):
    images = []
    images_count=0
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if args.video_angles == 1:
        out = cv2.VideoWriter('output_badangles.avi', fourcc, 5.0, (640, 240),0)
    else:
        out = cv2.VideoWriter('output_allangles.avi', fourcc, 15.0, (640, 240))
    with open('Ch2/interpolated.csv') as csvfile:
        if args.direction == 'center':
            _directions = ['center']
        elif args.direction == 'left':
            _directions = ['left']
        elif args.direction == 'left':
            _directions = ['right']
        else:
            _directions = ['center', 'left', 'right']

        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:

            d = row[5].split('/')[0]
            path = 'Ch2/' + d + '/' + row[5].split('/')[1]
            angle = float(row[6])
            torque = float(row[7])
            speed = float(row[8])

            if d in _directions:
                if d == 'center':
                    ix = 0
                elif d == 'left':
                    ix = 1
                else:
                    ix = 2
                #if images_count > 100:
                #    break
                images_count = images_count + 1

                #angle = correct_steering(angle, ix, args.steer_correction)
                #item = (path, angle, speed, torque, ix, 0)
                #img = pil_loader(path)
                #frame = np.asarray(img)
                #frame=np.load(path)
                if args.video_angles==1:

                    if angle*57.2958 <-30  or  angle*57.2958 >30  or angle==0  :
                        frame=cv2.imread(path)
                        frame=preprocess_image(frame)

                        if args.vide_pred==1:
                            angle_pred=model.predict(frame.reshape(-1, 240, 640, 3))[0][0]
                            cv2.putText(frame, str(angle_pred*57.2958), (300, 70), font, 1.0, (255, 0, 0), 3)
                        #cv2.putText(frame, str(angle), (300,10), font, 1.0, (0, 0, 255), 3)
                        #cv2.putText(frame, str(angle*57.2958), (300, 40), font, 1.0, (0, 0, 255), 3)
                        ##frame =cv2.imread(path)
                        out.write(frame)

                else:
                    frame = cv2.imread(path)
                    if args.vide_pred == 1:
                        angle_pred = model.predict(frame.reshape(-1, 480, 640, 3))[0][0]
                        cv2.putText(frame, str(angle_pred * 57.2958), (300, 120), font, 1.0, (255, 0, 0), 3)
                    cv2.putText(frame, str(angle), (300, 50), font, 1.0, (0, 0, 255), 3)
                    cv2.putText(frame, str(angle * 57.2958), (300, 80), font, 1.0, (0, 0, 255), 3)
                    ##frame =cv2.imread(path)
                    out.write(frame)

                ##cv2.imshow('frame', frame)

    print("Ended Video")
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return

def split_dataset_random(args):
    images = []
    with open('Ch2/interpolated.csv') as csvfile:
        if args.direction == 'center':
            _directions = ['center']
        elif args.direction == 'left':
            _directions = ['left']
        elif args.direction == 'left':
            _directions = ['right']
        else:
            _directions = ['center', 'left', 'right']

        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            d = row[5].split('/')[0]
            path = 'Ch2/' + d + '/' + row[5].split('/')[1]
            angle = float(row[6])
            torque = float(row[7])
            speed = float(row[8])

            if args.bad_angles==True:
                if angle==0:
                    continue
                if angle*57 > 50:
                    continue
                if angle*57 < -50:
                    continue

            if d in _directions:
                if d == 'center':
                    ix = 0
                elif d == 'left':
                    ix = 1
                else:
                    ix = 2
                #angle = correct_steering(angle, ix, args.steer_correction)
                item = (path, angle, speed, torque, ix, 0)
                images.append(item)
                if args.augmentation:
                    item = (path, -1 * angle, speed, torque, ix, 1)
                    images.append(item)

    #images = preprocess(images, args.steer_threshold, args.drop_low_angles)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2)

    return train_imgs, val_imgs
def generator_dataset_random(samples, args):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, args.batch_size):
            # print("Getting next batch")
            batch_samples = samples[offset:offset+args.batch_size]

            images = []
            y = []
            for bs in batch_samples:
                img_name  = bs[0]
                steering = bs[1]
                speed = bs[2]
                torque = bs[3]
                flip = bs[5]
                img = pil_loader(img_name)
                img = img if flip==0 else img.transpose(Image.FLIP_LEFT_RIGHT)  #cv2.flip(img,1)# Flip image if flip was 1
                img = np.asarray(img)
                #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
                #img = ((img - (255.0 / 2)) / 255.0)
                if args.augmentation:
                    img, steering = augment_image(img, steering)

                if args.output != 'angle':
                    y.append([steering, speed, torque])
                else:
                    y.append(steering)

                images.append(img)

            X_train = np.array(images)
            y_train = np.array(y)
            yield shuffle(X_train, y_train)
def generator_dataset_just_ahead(labels, index_values, args, scale=1.0, random_flip=False, input_shape=(240, 640,3)):

    batch_features = np.zeros((args.batch_size, *input_shape))
    batch_labels = np.zeros((args.batch_size, 1))
    num_samples=len(labels)-args.lookahead_window -1
    value_range = np.arange(0,len(labels)-args.lookahead_window-1)
    value_shuffle=shuffle(value_range)
    while True:
        #next_indexes = np.random.choice(np.arange(0, len(index_values) - 2*args.num_frames - args.lookahead_window + - 1), args.batch_size)
        for offset in range(0, num_samples, args.batch_size):
            index_to_batch=value_shuffle[offset:offset+args.batch_size]
            for i, idx in enumerate(index_to_batch):

                y = float(labels[idx])

                img_name=index_values[idx+args.lookahead_window]
                image = cv2.imread(img_name)
                #image = np.asarray(image)
                #image=preprocess_image(image)
                image=crop(image)
                image = np.asarray(image)

                batch_features[i, :] = image
                batch_labels[i] =y

            #batch_features.reshape((args.batch_size, args.num_frames, *input_shape))
            yield shuffle(batch_features,batch_labels)


def split_seq_dataset_chunks(args):
    images = []
    with open('Ch2/interpolated.csv') as csvfile:
        if args.direction == 'center':
            _directions = ['center']
        elif args.direction == 'left':
            _directions = ['left']
        elif args.direction == 'left':
            _directions = ['right']
        else:
            _directions = ['center', 'left', 'right']

        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            d = row[5].split('/')[0]
            path = 'Ch2/' + d + '/' + row[5].split('/')[1]
            angle = float(row[6])
            torque = float(row[7])
            speed = float(row[8])

            if d in _directions:
                if d == 'center':
                    ix = 0
                elif d == 'left':
                    ix = 1
                else:
                    ix = 2
                item = (path, angle, speed, torque, ix, 0)
                images.append(item)

    chunk_size = int(len(images) / 5)
    train_chunk = int(chunk_size * 0.8)
    print(chunk_size, train_chunk)

    train_imgs = []
    val_imgs = []

    for ix in range(0, len(images), chunk_size):
        chunk = images[ix:ix + chunk_size]
        train_imgs.extend(chunk[:train_chunk])
        val_imgs.extend(chunk[train_chunk:])

    return np.array(train_imgs), np.array(val_imgs)
def split_seq_dataset_simple(args):
    images = []
    with open('Ch2/interpolated.csv') as csvfile:
        if args.direction == 'center':
            _directions = ['center']
        elif args.direction == 'left':
            _directions = ['left']
        elif args.direction == 'left':
            _directions = ['right']
        else:
            _directions = ['center', 'left', 'right']

        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            d = row[5].split('/')[0]
            path = 'Ch2/' + d + '/' + row[5].split('/')[1]
            angle = float(row[6])
            torque = float(row[7])
            speed = float(row[8])

            if args.bad_angles == True:
                if angle == 0:
                    continue
                if angle * 57 > 50:
                    continue
                if angle * 57 < -50:
                    continue


            if args.angles_degree == True:
                angle=angle*radian_to_degree

            if d in _directions:
                if d == 'center':
                    ix = 0
                elif d == 'left':
                    ix = 1
                else:
                    ix = 2

                if args.use_more_big_angles == True:
                    if angle * 57 > 30 or angle * 57 < -30 :
                        for i in range(0,5):
                            item = (path, angle, speed, torque, ix, 0)
                            images.append(item)
                item = (path, angle, speed, torque, ix, 0)
                images.append(item)

    images_len = int(len(images))
    train_chunk = int(images_len * 0.8)
    print(images_len, train_chunk)

    train_imgs = []
    val_imgs = []

    train_imgs.extend(images[:train_chunk])
    val_imgs.extend(images[train_chunk:])

    return np.array(train_imgs), np.array(val_imgs)
def generator_seq_dataset_chunks(labels, index_values, args, scale=1.0, random_flip=False, input_shape=(480, 640, 3)):
    batch_features = np.zeros((args.batch_size, args.num_frames, *input_shape))
    batch_labels = np.zeros((args.batch_size, 1))
    value_range = np.arange(0,len(labels)-args.num_frames-1)
    while True:
        next_indexes = np.random.choice(np.arange(0, len(index_values) - args.num_frames - 1), args.batch_size)
        for i, idx in enumerate(next_indexes):
            for j in range(args.num_frames):
                y = labels[idx+j]
                #image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx+j])))
                #image = np.load(image_path)
                img_name=index_values[idx+j]
                image = cv2.imread(img_name)
                image = np.asarray(image)

                if random_flip:
                    flip_bit = random.randint(0, 1)
                    if flip_bit == 1:
                        image = np.flip(image, 1)
                        y = y * -1
                #image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                #image = ((image - (255.0 / 2)) / 255.0)
                batch_features[i, j, :] = image
                batch_labels[i] = y * scale


        yield batch_features, batch_labels
def generator_seq_dataset_chunks_ahead(labels, index_values, args, scale=1.0, random_flip=False, input_shape=(480, 640, 3)):

    batch_features = np.zeros((args.batch_size, 2*args.num_frames, *input_shape))
    batch_labels = np.zeros((args.batch_size, 1))
    num_samples=len(labels)-2*args.num_frames - args.lookahead_window-1
    value_range = np.arange(0,len(labels)-2*args.num_frames - args.lookahead_window-1)
    value_shuffle=shuffle(value_range)
    while True:
        #next_indexes = np.random.choice(np.arange(0, len(index_values) - 2*args.num_frames - args.lookahead_window + - 1), args.batch_size)
        for offset in range(0, num_samples, args.batch_size):
            index_to_batch=value_shuffle[offset:offset+args.batch_size]
            for i, idx in enumerate(index_to_batch):
                for j in range(args.num_frames):
                    y = float(labels[idx+j])

                    img_name=index_values[idx+j]
                    image = cv2.imread(img_name)
                    image = np.asarray(image)

                    if random_flip:
                        flip_bit = random.randint(0, 1)
                        if flip_bit == 1:
                            image = np.flip(image, 1)
                            y = y * -1
                    #image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                    #image = ((image - (255.0 / 2)) / 255.0)
                    batch_features[i, j, :] = image
                    batch_labels[i] = y

                for j in range(args.num_frames):
                    img_name_ahead=index_values[idx+j+args.lookahead_window]
                    image_ahead = cv2.imread(img_name_ahead)
                    image_ahead = np.asarray(image_ahead)
                    #image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                    #image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                    batch_features[i, args.num_frames+j, :] = image_ahead


            yield batch_features, batch_labels

def generator_seq_dataset_chunks_ahead_v2(labels, index_values, args, scale=1.0, random_flip=False, input_shape=(480, 640, 3)):

    batch_features = np.zeros((args.batch_size, 2*args.num_frames, *input_shape))
    batch_labels = np.zeros((args.batch_size, 1))
    num_samples=len(labels)-args.num_frames - args.lookahead_window-1
    value_range = np.arange(0,len(labels)-args.num_frames - args.lookahead_window-1)
    value_shuffle=shuffle(value_range)
    while True:
        #next_indexes = np.random.choice(np.arange(0, len(index_values) - 2*args.num_frames - args.lookahead_window + - 1), args.batch_size)
        for offset in range(0, num_samples, args.batch_size):
            index_to_batch=value_shuffle[offset:offset+args.batch_size]
            for i, idx in enumerate(index_to_batch):
                for j in range(args.num_frames):
                    y = float(labels[idx+j])

                    img_name=index_values[idx+j]
                    image = cv2.imread(img_name)
                    image = np.asarray(image)

                    if random_flip:
                        flip_bit = random.randint(0, 1)
                        if flip_bit == 1:
                            image = np.flip(image, 1)
                            y = y * -1
                    #image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                    #image = ((image - (255.0 / 2)) / 255.0)
                    batch_features[i, j, :] = image
                    batch_labels[i] = y

                for j in range(args.num_frames):
                    img_name_ahead=index_values[idx-j+args.lookahead_window]
                    image_ahead = cv2.imread(img_name_ahead)
                    image_ahead = np.asarray(image_ahead)
                    #image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                    #image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                    batch_features[i, args.num_frames+j, :] = image_ahead


            yield batch_features, batch_labels


def generator_seq_dataset_chunks_ahead_time(labels, index_values, args, scale=1.0, random_flip=False, input_shape=(240, 640, 3)):
    batch_features = np.zeros((args.batch_size, 2, args.num_frames, *input_shape))
    batch_labels = np.zeros((args.batch_size, 1))
    num_samples = len(labels) - 2 * args.num_frames - args.lookahead_window - 1
    value_range = np.arange(0, len(labels) - 2 * args.num_frames - args.lookahead_window - 1)
    value_shuffle = shuffle(value_range)
    while True:
        # next_indexes = np.random.choice(np.arange(0, len(index_values) - 2*args.num_frames - args.lookahead_window + - 1), args.batch_size)
        for offset in range(0, num_samples, args.batch_size):
            index_to_batch = value_shuffle[offset:offset + args.batch_size]
            for i, idx in enumerate(index_to_batch):
                for j in range(args.num_frames):
                    y = float(labels[idx + j])

                    img_name = index_values[idx + j]
                    image = cv2.imread(img_name)
                    image=crop(image)
                    image = np.asarray(image)

                    if random_flip:
                        flip_bit = random.randint(0, 1)
                        if flip_bit == 1:
                            image = np.flip(image, 1)
                            y = y * -1
                    # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                    # image = ((image - (255.0 / 2)) / 255.0)
                    batch_features[i,0, j, :] = image
                    batch_labels[i] = y

                for j in range(args.num_frames):
                    img_name_ahead = index_values[idx + j + args.lookahead_window]
                    image_ahead = cv2.imread(img_name_ahead)
                    image_ahead = crop(image_ahead)
                    image_ahead = np.asarray(image_ahead)
                    # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                    # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                    batch_features[i,1,j, :] = image_ahead

            yield batch_features, batch_labels


def generator_seq_dataset_just_ahead(labels, index_values, args, scale=1.0, random_flip=False, input_shape=(240, 640,1)):

    batch_features = np.zeros((args.batch_size, args.num_frames, *input_shape))
    batch_labels = np.zeros((args.batch_size, 1))
    num_samples=len(labels)-args.num_frames -1
    value_range = np.arange(0,len(labels)-args.num_frames-1)
    value_shuffle=shuffle(value_range)
    while True:
        #next_indexes = np.random.choice(np.arange(0, len(index_values) - 2*args.num_frames - args.lookahead_window + - 1), args.batch_size)
        for offset in range(0, num_samples, args.batch_size):
            index_to_batch=value_shuffle[offset:offset+args.batch_size]
            for i, idx in enumerate(index_to_batch):
                for j in range(args.num_frames):
                    y = float(labels[idx+j])

                    img_name=index_values[idx+j]
                    image = cv2.imread(img_name)
                    #image = np.asarray(image)
                    image=preprocess_image(image)
                    image = np.asarray(image)

                    image=image.reshape(*input_shape)
                    if random_flip:
                        flip_bit = random.randint(0, 1)
                        if flip_bit == 1:
                            image = np.flip(image, 1)
                            y = y * -1
                    #image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                    #image = ((image - (255.0 / 2)) / 255.0)
                    batch_features[i, j, :] = image
                batch_labels[i] =float(labels[idx])

            #batch_features.reshape((args.batch_size, args.num_frames, *input_shape))
            yield batch_features, batch_labels
def generator_seq_dataset_diff_ahead(labels, index_values, args, scale=1.0, random_flip=False, input_shape=(240, 640, 3)):

    batch_features = np.zeros((args.batch_size, 240*(args.num_frames), 640, 3))
    batch_labels = np.zeros((args.batch_size, 1))
    num_samples=len(labels)-args.num_frames - args.lookahead_window-1
    value_range = np.arange(0,len(labels)-args.num_frames - args.lookahead_window-1)
    value_shuffle=shuffle(value_range)
    while True:
        #next_indexes = np.random.choice(np.arange(0, len(index_values) - 2*args.num_frames - args.lookahead_window + - 1), args.batch_size)
        for offset in range(0, num_samples, args.batch_size):
            index_to_batch=value_shuffle[offset:offset+args.batch_size]
            for i, idx in enumerate(index_to_batch):

                y = float(labels[idx])
                img_name = index_values[idx]
                image = cv2.imread(img_name)
                image = crop(image)
                image = np.asarray(image)
                concat_image=image

                if args.mode == 'diff':
                    img_name_ahead = index_values[idx + args.lookahead_window]
                    image_ahead = cv2.imread(img_name_ahead)
                    image_ahead = crop(image_ahead)
                    image_ahead = np.asarray(image_ahead)
                    # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                    # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                    concat_image = image - image_ahead
                    for j in range(args.num_frames-1):
                        img_name_ahead = index_values[idx + j +1 + args.lookahead_window]
                        image_ahead = cv2.imread(img_name_ahead)
                        image_ahead = crop(image_ahead)
                        image_ahead = np.asarray(image_ahead)
                        # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                        # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                        diff=image-image_ahead
                        concat_image=np.concatenate((concat_image, diff), axis=0)
                if args.mode == 'concat':
                    for j in range(args.num_frames-1):
                        img_name_ahead = index_values[idx + j + args.lookahead_window]
                        image_ahead = cv2.imread(img_name_ahead)
                        image_ahead = crop(image_ahead)
                        image_ahead = np.asarray(image_ahead)
                        # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                        # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                        concat_image=np.concatenate((concat_image, image_ahead), axis=0)
                batch_features[i, :] = concat_image
                batch_labels[i] = y

            yield batch_features, batch_labels

def split_seq_dataset_groups(args):
    images = []

    df = pd.read_csv("Ch2/interpolated.csv")
    for row in df.iterrows():
        angle = row[1]['angle']
        fname = "Ch2/{}".format(row[1]['filename'])
        if 'center' in fname:
            path = os.path.join(args.root_dir, fname)
            item = (path, angle)
            images.append(item)

    lookahead_window = 40

    seq_images = []
    for ix, im in enumerate(images[:-lookahead_window]):
         if ix < args.window_len-1:
             continue
         fim = [images[ix+lookahead_window], images[ix]]
         #print("a",images[ix+lookahead_window])
         #print("c",images[ix])
         if args.lookahead == True:
             seq_images.append(fim)
         else:
             seq_images.append(images[ix-args.window_len+1:ix+1])

    chunk_size = int(len(seq_images)/5)
    train_chunk = int(chunk_size*0.8)
    print(chunk_size, train_chunk)

    train_imgs = []
    val_imgs = []

    for ix in range(0,len(seq_images), chunk_size):
        chunk = seq_images[ix:ix+chunk_size]
        train_imgs.extend(chunk[:train_chunk])
        val_imgs.extend(chunk[train_chunk:])

    #seq_images = preprocess(seq_images, steer_threshold, drop_low_angles)
    return np.array(train_imgs), np.array(val_imgs)
def generator_seq_dataset_groups(samples, args):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, args.batch_size):
            # print("Getting next batch")
            batch_samples = samples[offset:offset+args.batch_size]

            seq_images = []
            y = []
            for bs in batch_samples:
                images = []
                outs = None
                for bi in bs:
                    img_name  = bi[0]
                    steering = np.float32(bi[1])
                    img = cv2.imread(img_name)
                    if args.greyscale:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.asarray(img)
                    img = img[:,:,np.newaxis]
                    #img = img - 127
                    images.append(img)
                    outs = steering
                #print("Raw images shape: ", np.asarray(images).shape)
                if args.mode == 'diff':
                    imdiff = [im1-im2 for im1,im2 in zip(images[1:], images[:-1])]
                    #print("Diff images shape: ", np.asarray(imdiff).shape)
                    seq_images.append(np.concatenate(imdiff, axis=2))
                if args.mode == 'concat':
                    seq_images.append(np.concatenate(images, axis=2)) #images[0]-images[1])
                #print(np.mean(seq_images[-1]))
                y.append(outs)

            X_train = np.array(seq_images)
            y_train = np.array(y)
            yield shuffle(X_train, y_train)

def predict(model,pretrained_weights,args):
    model.load_weights("{}.h5".format(pretrained_weights))
    #print(model.get_weights())
    print("Start the predictions")
    pred = []
    real = []
    df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float, 'torque': np.float, 'speed': np.float})
    df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

    fnames = np.array('Ch2/' + df['filename'])
    angles = np.array(df['angle'])
    for ix, tr in enumerate(zip(fnames, angles)):
        if  args.num_samples > 0:
            if ix >= args.num_samples:
               break
        img, angle = val_output(tr)

        if args.bad_angles == True:
            if angle == 0:
                continue
            if angle * 57 > 50:
                continue
            if angle * 57 < -50:
                continue

        real.append(angle)
        prediction=model.predict(img.reshape(-1, 480, 640, 3))[0][0]
        print(prediction)
        pred.append(prediction)
        #if ix % 1000 == 0:
        #    print(str(ix))
        #if ix> 1000:
        #   break

    pred = np.array(pred)

    real = np.array(real)
    error=pred-real
    print(error)
    print("Mean Error: ", np.sqrt(np.mean((pred - real) ** 2)))
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights + 'predictions.png')

    plt.figure(figsize=(16, 9))
    plt.plot(error, label='Error')
    plt.legend()
    plt.savefig(pretrained_weights + 'error.png')
    # plt.show()

    # df = pd.DataFrame()
    # df['fnames'] = fnames
    # df['angle'] =  real
    # df['pred'] = pred
    # df.to_csv('results_lookahead.csv')
def predict_temporal(model,pretrained_weights,args, input_shape=(480, 640, 3)):
    model.load_weights("{}.h5".format(pretrained_weights))
    #print(model.get_weights())

    temporal_features = np.zeros(( 2 * args.num_frames, *input_shape))


    images = load_images(args)
    pred = []
    real = []
    #df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float, 'torque': np.float, 'speed': np.float})
    #df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

    index_values=images[:, 0]
    labels=(images[:, 1])

    num_samples=len(labels)-2*args.num_frames - args.lookahead_window-1

    for index in range(0,num_samples):
        angle =float(labels[index])

        if args.bad_angles == True:
            if angle == 0:
                continue
            if angle * 57 > 50:
                continue
            if angle * 57 < -50:
                continue

        for j in range(args.num_frames):
            temporal_label = float(labels[index + j])

            if args.angles_degree == True:
                temporal_label = temporal_label * radian_to_degree

            img_name = index_values[index + j]
            image = cv2.imread(img_name)
            image = np.asarray(image)

            # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            # image = ((image - (255.0 / 2)) / 255.0)
            temporal_features[j, :] = image


        for j in range(args.num_frames):
            img_name_ahead = index_values[index + j + args.lookahead_window]
            image_ahead = cv2.imread(img_name_ahead)
            image_ahead = np.asarray(image_ahead)
            # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
            # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
            temporal_features[args.num_frames + j, :] = image_ahead


        real.append(temporal_label)
        prediction=model.predict(temporal_features.reshape(-1,args.num_frames*2,480,640,3))[0][0]
        pred.append(float(prediction))
        if index % 1000 == 0:
            print(str(index))
        #if index> 100:
        #   break

    pred = np.array(pred)
    real = np.array(real)
    print("Mean Error: ", np.sqrt(np.mean((pred - real) ** 2)))
    if args.angles_degree == True:
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred - real) ** 2))/radian_to_degree)
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred/radian_to_degree - real/radian_to_degree) ** 2)) )
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights + 'predictions_temporal.png')


def predict_temporalv2(model,pretrained_weights,args, input_shape=(480, 640, 3)):
    model.load_weights("{}.h5".format(pretrained_weights))
    #print(model.get_weights())

    temporal_features = np.zeros(( 2 * args.num_frames, *input_shape))


    images = load_images(args)
    pred = []
    real = []
    #df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float, 'torque': np.float, 'speed': np.float})
    #df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

    index_values=images[:, 0]
    labels=(images[:, 1])

    num_samples=len(labels)-args.num_frames - args.lookahead_window-1

    for index in range(0,num_samples):
        angle =float(labels[index])

        if args.bad_angles == True:
            if angle == 0:
                continue
            if angle * 57 > 50:
                continue
            if angle * 57 < -50:
                continue

        for j in range(args.num_frames):
            temporal_label = float(labels[index + j])

            if args.angles_degree == True:
                temporal_label = temporal_label * radian_to_degree

            img_name = index_values[index + j]
            image = cv2.imread(img_name)
            image = np.asarray(image)

            # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            # image = ((image - (255.0 / 2)) / 255.0)
            temporal_features[j, :] = image


        for j in range(args.num_frames):
            img_name_ahead = index_values[index - j + args.lookahead_window]
            image_ahead = cv2.imread(img_name_ahead)
            image_ahead = np.asarray(image_ahead)
            # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
            # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
            temporal_features[args.num_frames + j, :] = image_ahead


        real.append(temporal_label)
        prediction=model.predict(temporal_features.reshape(-1,args.num_frames*2,480,640,3))[0][0]
        pred.append(float(prediction))
        if index % 1000 == 0:
            print(str(index))
        #if index> 100:
        #   break

    pred = np.array(pred)
    real = np.array(real)
    print("Mean Error: ", np.sqrt(np.mean((pred - real) ** 2)))
    if args.angles_degree == True:
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred - real) ** 2))/radian_to_degree)
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred/radian_to_degree - real/radian_to_degree) ** 2)) )
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights + 'predictions_temporal.png')

def predict_temporal_timedistrib(model,pretrained_weights,args, input_shape=(240, 640, 3)):
    model.load_weights("{}.h5".format(pretrained_weights))
    #print(model.get_weights())

    temporal_features = np.zeros(( 2, args.num_frames, *input_shape))


    images = load_images(args)
    pred = []
    real = []
    #df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float, 'torque': np.float, 'speed': np.float})
    #df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

    index_values=images[:, 0]
    labels=(images[:, 1])

    num_samples=len(labels)-2*args.num_frames - args.lookahead_window-1

    for index in range(0,num_samples):
        angle =float(labels[index])

        if args.bad_angles == True:
            if angle == 0:
                continue
            if angle * 57 > 50:
                continue
            if angle * 57 < -50:
                continue

        for j in range(args.num_frames):
            temporal_label = float(labels[index + j])

            if args.angles_degree == True:
                temporal_label = temporal_label * radian_to_degree

            img_name = index_values[index + j]
            image = cv2.imread(img_name)
            image=crop(image)
            image = np.asarray(image)

            # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            # image = ((image - (255.0 / 2)) / 255.0)
            temporal_features[0,j, :] = image


        for j in range(args.num_frames):
            img_name_ahead = index_values[index + j + args.lookahead_window]
            image_ahead = cv2.imread(img_name_ahead)
            image_ahead=crop(image_ahead)
            image_ahead = np.asarray(image_ahead)
            # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
            # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
            temporal_features[1,j, :] = image_ahead


        real.append(temporal_label)
        prediction=model.predict(temporal_features.reshape(-1,args.num_frames*2,480,640,3))[0][0]
        pred.append(float(prediction))
        if index % 1000 == 0:
            print(str(index))
        #if index> 100:
        #   break

    pred = np.array(pred)
    real = np.array(real)
    print("Mean Error: ", np.sqrt(np.mean((pred - real) ** 2)))
    if args.angles_degree == True:
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred - real) ** 2))/radian_to_degree)
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred/radian_to_degree - real/radian_to_degree) ** 2)) )
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights + 'predictions_temporal.png')
def predict_diff(model,pretrained_weights,args, input_shape=(240, 640, 3)):
    model.load_weights("{}.h5".format(pretrained_weights))
    #print(model.get_weights())

    temporal_features = np.zeros(( 240*(args.num_frames), 640, 3))


    images = load_images(args)
    pred = []
    real = []
    #df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float, 'torque': np.float, 'speed': np.float})
    #df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

    index_values=images[:, 0]
    labels=(images[:, 1])

    num_samples=len(labels)-args.num_frames - args.lookahead_window-1

    for index in range(0,num_samples):
        angle =float(labels[index])

        if args.bad_angles == True:
            if angle == 0:
                continue
            if angle * 57 > 50:
                continue
            if angle * 57 < -50:
                continue
        y = float(labels[index])
        img_name = index_values[index]
        image = cv2.imread(img_name)
        image = crop(image)
        image = np.asarray(image)
        concat_image = image

        if args.mode == 'diff':
            img_name_ahead = index_values[index + args.lookahead_window]
            image_ahead = cv2.imread(img_name_ahead)
            image_ahead = crop(image_ahead)
            image_ahead = np.asarray(image_ahead)
            # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
            # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
            concat_image = image - image_ahead
            for j in range(args.num_frames-1):
                img_name_ahead = index_values[index + j + 1 + args.lookahead_window]
                image_ahead = cv2.imread(img_name_ahead)
                image_ahead = crop(image_ahead)
                image_ahead = np.asarray(image_ahead)
                # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                diff = image - image_ahead
                concat_image = np.concatenate((concat_image, diff), axis=0)
        if args.mode == 'concat':
            for j in range(args.num_frames - 1):
                img_name_ahead = index_values[index + j + args.lookahead_window]
                image_ahead = cv2.imread(img_name_ahead)
                image_ahead = crop(image_ahead)
                image_ahead = np.asarray(image_ahead)
                # image_ahead[:, :, 0] = cv2.equalizeHist(image_ahead[:, :, 0])
                # image_ahead = ((image_ahead - (255.0 / 2)) / 255.0)
                concat_image = np.concatenate((concat_image, image_ahead), axis=0)

        temporal_features = concat_image
        temporal_label = y


        real.append(temporal_label)
        prediction=model.predict(temporal_features.reshape(-1,args.num_frames *240,640,3))[0][0]
        pred.append(float(prediction))
        if index % 1000 == 0:
            print(str(index))
        #if index> 1:
        #  break

    pred = np.array(pred)
    real = np.array(real)
    rmsenum= np.sqrt(np.mean((pred - real) ** 2))
    print("Mean Error: ",rmsenum)
    if args.angles_degree == True:
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred - real) ** 2))/radian_to_degree)
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred/radian_to_degree - real/radian_to_degree) ** 2)) )
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights + 'predictions_temporal.png')

    return rmsenum

def predict_just_ahead(model,pretrained_weights,args, input_shape=(240, 640,1)):
    model.load_weights("{}.h5".format(pretrained_weights))
    #print(model.get_weights())

    temporal_features = np.zeros((args.num_frames, *input_shape))


    images = load_images(args)
    pred = []
    real = []
    #df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float, 'torque': np.float, 'speed': np.float})
    #df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

    index_values=images[:, 0]
    labels=(images[:, 1])

    num_samples=len(labels)-args.num_frames -1

    for index in range(0,num_samples):
        angle =float(labels[index])

        if args.bad_angles == True:
            if angle == 0:
                continue
            if angle * 57 > 50:
                continue
            if angle * 57 < -50:
                continue

        for j in range(args.num_frames):


            if args.angles_degree == True:
                temporal_label = temporal_label * radian_to_degree

            img_name = index_values[index + j]
            image = cv2.imread(img_name)
            image = np.asarray(image)
            image=preprocess_image(image)

            # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            # image = ((image - (255.0 / 2)) / 255.0)
            temporal_features[j, :] = image
        temporal_label = float(labels[index])



        real.append(temporal_label)
        prediction=model.predict(temporal_features.reshape(-1,args.num_frames,240,640,1))[0][0]
        pred.append(float(prediction))
        if index % 1000 == 0:
            print(str(index))
        #if index> 100:
        #   break

    pred = np.array(pred)
    real = np.array(real)
    print("Mean Error: ", np.sqrt(np.mean((pred - real) ** 2)))
    if args.angles_degree == True:
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred - real) ** 2))/radian_to_degree)
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred/radian_to_degree - real/radian_to_degree) ** 2)) )
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights + 'predictions_temporal.png')
def predict_shift(model,pretrained_weights,args):
    model.load_weights("{}.h5".format(pretrained_weights))
    pred = []
    real = []
    full_imgs, fnames = make_full_dataset(args.root_dir, args.direction,
                                          True, args.steer_threshold, args.drop_low_angles,
                                          args.steer_correction, args.window_len, args.lookahead)
    full_gen = generator_seq_dataset_groups(full_imgs, args)

    print("Full imgs shape: ", full_imgs.shape)
    for ix in range(int(len(full_imgs) / args.batch_size)):
        if args.num_samples > 0:
            if ix >= (args.num_samples / args.batch_size):
                break

        inp, _real = next(full_gen)
        _pred = model.predict(inp)
        # print(_pred[:,0].shape, _real.shape)
        real.extend(_real)
        pred.extend(_pred[:, 0])
        if ix % 100 == 0:
            print(str(ix))

    pred = np.array(pred)
    real = np.array(real)
    print("Mean Error: ", np.sqrt(np.mean((pred - real) ** 2)))
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights+ 'pred_baseline.png')
    # plt.show()
    #print(len(fnames), pred.shape, real.shape)
    #df = pd.DataFrame()
    #df['fnames'] = fnames[:len(real)]
    #df['angle'] = real
    #df['pred'] = pred
    #df.to_csv('results_baseline.csv')

    #print(history_object.history['val_loss'])
    #print(history_object.history['loss'])


def predict_just_ahead_nvidia(model,pretrained_weights,args, input_shape=(240, 640,3)):
    model.load_weights("{}.h5".format(pretrained_weights))
    #print(model.get_weights())

    temporal_features = np.zeros(input_shape)


    images = load_images(args)
    pred = []
    real = []
    #df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float, 'torque': np.float, 'speed': np.float})
    #df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

    index_values=images[:, 0]
    labels=(images[:, 1])

    num_samples=len(labels)-args.lookahead_window -1

    for index in range(0,num_samples):
        angle =float(labels[index])

        if args.bad_angles == True:
            if angle == 0:
                continue
            if angle * 57 > 50:
                continue
            if angle * 57 < -50:
                continue


        if args.angles_degree == True:
            temporal_label = temporal_label * radian_to_degree

        img_name = index_values[index + args.lookahead_window]
        image = cv2.imread(img_name)
        image = crop(image)
        image = np.asarray(image)


        # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        # image = ((image - (255.0 / 2)) / 255.0)
        temporal_features[:] = image
        temporal_label = float(labels[index])



        real.append(temporal_label)
        prediction=model.predict(temporal_features.reshape(-1,240,640,3))[0][0]
        pred.append(float(prediction))
        #if index % 1000 == 0:
        #    print(str(index))
        #if index> 100:
        #   break

    pred = np.array(pred)
    real = np.array(real)
    print("Mean Error: ", np.sqrt(np.mean((pred - real) ** 2)))
    if args.angles_degree == True:
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred - real) ** 2))/radian_to_degree)
        print("Mean Error in  radians: ", np.sqrt(np.mean((pred/radian_to_degree - real/radian_to_degree) ** 2)) )
    plt.figure(figsize=(16, 9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig(pretrained_weights + 'predictions_temporal.png')

# utils functions

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# Save histogram of steering angles
def save_hist(data, name):
    plt.figure()
    plt.hist(data, bins=20, color='green')
    plt.xlabel('Steering angles')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of '+name.replace("_"," "))
    plt.savefig(name+".png")

def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    cols, rows = img_shape
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image_tr, steer_ang

def augment_image(im, angle):
    im = augment_brightness(im)
    #im, angle = trans_image(im, angle, 50)
    return im, angle

def target_normalization(imgs):

    print("Angles Before Normalization: ", np.mean(imgs[:,1]),  np.std(imgs[:,1]))
    print("Speed Before Normalization: ", np.mean(imgs[:,2]),  np.std(imgs[:,2]))
    print("Torque Before Normalization: ", np.mean(imgs[:,3]),  np.std(imgs[:,3]))

    means = [-1.5029124843010954e-05, 15.280215684662938, -0.09274196373527277]
    stds = [0.338248459692, 5.5285360815, 0.905531102853]
    imgs[:,1] = (imgs[:,1]-means[0])/stds[0]
    imgs[:,2] = (imgs[:,2]-means[1])/stds[1]
    imgs[:,3] = (imgs[:,3]-means[2])/stds[2]

    return imgs

def val_output(s):
    img_name  = s[0]
    steering = s[1]
    img = pil_loader(img_name)
    img = np.asarray(img)
    return img, steering

def angle_loss(y_true, y_pred):
    return K.mean((y_pred[:,0]-y_true[:,0])**2)

def make_full_dataset(dir, direction, train, steer_threshold, drop_low_angles, steer_correction, window_len, lookahead):
    images = []
    fnames = []
    df = pd.read_csv("Ch2/interpolated.csv")
    for row in df.iterrows():
        angle = row[1]['angle']
        fname = "Ch2/{}".format(row[1]['filename'])
        if 'center' in fname:
            fnames.append(fname)
            path = os.path.join(dir, fname)
            item = (path, angle)
            images.append(item)

    lookahead_window = 40

    seq_images = []
    for ix, im in enumerate(images[:-lookahead_window]):
         if ix < window_len-1:
             continue
         fim = [images[ix+lookahead_window], images[ix]]
         if lookahead == True:
             seq_images.append(fim)
         else:
             seq_images.append(images[ix-window_len+1:ix+1])

    return np.array(seq_images), fnames

def make_seq_dataset(args):
    images = []

    df = pd.read_csv("Ch2/interpolated.csv")
    for row in df.iterrows():
        angle = row[1]['angle']
        fname = "Ch2/{}".format(row[1]['filename'])
        if 'center' in fname:
            path = os.path.join(args.root_dir, fname)
            item = (path, angle)
            images.append(item)

    lookahead_window = 40

    seq_images = []
    for ix, im in enumerate(images[:-lookahead_window]):
         if ix < args.window_len-1:
             continue
         fim = [images[ix+lookahead_window], images[ix]]
         #print("a",images[ix+lookahead_window])
         #print("c",images[ix])
         if args.lookahead == True:
             seq_images.append(fim)
         else:
             seq_images.append(images[ix-args.window_len+1:ix+1])

    chunk_size = int(len(seq_images)/5)
    train_chunk = int(chunk_size*0.8)
    print(chunk_size, train_chunk)

    train_imgs = []
    val_imgs = []

    for ix in range(0,len(seq_images), chunk_size):
        chunk = seq_images[ix:ix+chunk_size]
        train_imgs.extend(chunk[:train_chunk])
        val_imgs.extend(chunk[train_chunk:])

    #seq_images = preprocess(seq_images, steer_threshold, drop_low_angles)
    return np.array(train_imgs), np.array(val_imgs)

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle

def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
#    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training, data_type = "unity3d"):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            if(data_type == "unity3d"):
                center, left, right = image_paths[index]
                steering_angle = steering_angles[index]
                # argumentation
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = augument(data_dir, center, left, right, steering_angle)
                else:
                    image = load_image(data_dir, center)
                # add the image and steering angle to the batch
                images[i] = preprocess(image)
                steers[i] = steering_angle
                i += 1
                if i == batch_size:
                    break
            elif(data_type == "udacity"):
                img_path = image_paths[index]
#                center = None
#                left = None
#                right = None
#                if(val == center):
#                    center = img_path
#                elif(val == left):
#                    left = img_path
#                elif(val == right):
#                    right = img_path
                steering_angle = steering_angles[index]
                image = load_image(data_dir, img_path)
                # add the image and steering angle to the batch
                images[i] = preprocess(image)
                steers[i] = steering_angle
                i += 1
                if i == batch_size:
                    break
        yield images, steers

def load_image(image_index, image_base_path, size):
    batch_value = np.zeros(size)
    for i in image_index:
        image_path = os.path.join(image_base_path, "{}.jpg.npy".format(i))
        image = np.load(image_path)
        batch_value[i] = image
    return batch_value

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def std_evaluate(model, generator, size):
    """
    """
    #size = generator.get_size()
    #batch_size = generator.get_batch_size()
    #n_batches = size // batch_size
    print("std test")

    err_sum = 0.
    err_count = 0.
    count = 0
    for data in generator:
        count += 1
        X_batch, y_batch = data
        y_pred = model.predict_on_batch(X_batch)
        err_sum += np.sum((y_batch - y_pred) ** 2)
        err_count += len(y_pred)
        if count > size-1:
            break

    mse = err_sum / err_count
    return [mse, np.sqrt(mse)]

def std_evaluate_seq(model, generator, size, seq_size):
    """
    """
    #size = generator.get_size()
    #batch_size = generator.get_batch_size()
    #n_batches = size // batch_size
    print("std test")

    err_sum = 0.
    err_count = 0.
    count = 0
    for data in generator:
        count += 1
        X_batch, y_batch = data
        y_pred = model.predict_on_batch(X_batch)
        err_sum += np.sum((y_batch - y_pred) ** 2)
        err_count += len(y_pred)
        if count == size:
            break

    mse = err_sum / err_count
    return [mse, np.sqrt(mse)]

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

