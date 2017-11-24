from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
from random import shuffle
import numpy as np
import os
import csv
from six.moves import cPickle
from keras.utils.io_utils import HDF5Matrix
import h5py
import math
from keras import metrics
from keras.models import model_from_json

def load_all_images(filenames, path):
    images = []
    for filename in filenames:
        filename = filename.split('.')[0]+'.png'
        print('Filename '+filename)
        im = cv2.resize(cv2.imread(os.path.join(path, filename)), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        images.append(im)
    images=np.array(images).reshape((-1,3,224,224))
    return images

def vgg_net():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid')) 
    return model

def load_images(path, division, labels):
    images = list()
    labels_img = list()
    count=0
    f=h5py.File('train.h5','w')
    X_dset = f.create_dataset('my_data',(division, 3,224,224), maxshape=(None,3,224,224), chunks=True, dtype='f')
    y_dset = f.create_dataset('my_labels',(division,1), maxshape=(None,1), chunks=True, dtype='i')
    countPositive=0
    countNegative=0
    posFiles = list()
    negFiles = list()
    for key_filename in labels:
        if labels[key_filename] == 1:
            posFiles.append(key_filename)
        else:
            negFiles.append(key_filename)
    shuffle(posFiles)
    shuffle(negFiles)
    minElements = min(len(posFiles),len(negFiles))
    print('minElements: %d'%minElements)
    filesList = [posFiles[i] for i in range(minElements)] + [negFiles[j] for j in range(minElements)]
    shuffle(filesList)
    #indexPartition = int(math.floor((0.666)*float(2*minElements)))
    indexPartition = int(2*minElements-1)
    trainFiles = filesList[0:indexPartition]
    testFiles = filesList[indexPartition:len(filesList)]
    print('indexPartition: %d ,len trainFiles: %d, len testFiles %d'%(indexPartition, len(trainFiles), len(testFiles)))
    for filename in trainFiles:
        im = cv2.resize(cv2.imread(os.path.join(path, filename)), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        images.append(im)
        labels_img.append(labels[filename])
        if labels[filename] == 1:
            countPositive = countPositive +1
        else:
            countNegative = countNegative +1
        count = count+1
        if count % division == 0 or count ==len(labels):
            X = np.array(images).reshape((-1,3,224,224))
            y = np.array(labels_img).reshape((-1,1))
            X_dset.resize((count,3,224,224))
            X_dset[count:,:,:,:] = X
            y_dset.resize((count,1))
            y_dset[count:,:]=y
            images=list()
            labels_img=list()
    print('train countPositive: %d'%countPositive)
    print('train countNegative: %d'%countNegative)
    f.close()
    f=h5py.File('train.h5','w')
    X_dset = f.create_dataset('my_data',(division, 3,224,224), maxshape=(None,3,224,224), chunks=True, dtype='f')
    y_dset = f.create_dataset('my_labels',(division,1), maxshape=(None,1), chunks=True, dtype='i')
    countPositive=0
    countNegative=0
    for filename in testFiles:
        im = cv2.resize(cv2.imread(os.path.join(path, filename)), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        images.append(im)
        labels_img.append(labels[filename])
        if labels[filename] == 1:
            countPositive = countPositive +1
        else:
            countNegative = countNegative +1
        count = count+1
        if count % division == 0 or count ==len(labels):
            X = np.array(images).reshape((-1,3,224,224))
            y = np.array(labels_img).reshape((-1,1))
            X_dset.resize((count,3,224,224))
            X_dset[count:,:,:,:] = X
            y_dset.resize((count,1))
            y_dset[count:,:]=y
            images=list()  
            labels_img=list()
    print('test countPositive: %d'%countPositive)
    print('test countNegative: %d'%countNegative)
    f.close()

def load_images_test(path, division):
    images = list()  
    img_loc = list()
    count=0
    f=h5py.File('test.h5','w')
    X_dset = f.create_dataset('my_data',(division, 3,224,224), maxshape=(None,3,224,224), chunks=True, dtype='f')
    for filename in os.listdir(path):
        im = cv2.resize(cv2.imread(os.path.join(path, filename)), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        images.append(im)
	img_loc.append(filename)  
        count = count +1 
        if count % division == 0:
            X = np.array(images).reshape((-1,3,224,224))
            X_dset.resize((count,3,224,224)) 
            X_dset[count:,:,:,:] = X
            images=list()  
    f.close()
    return img_loc


def load_labels(csv_path):
    labels=dict()
    csvfile =  open(csv_path, 'rb')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        filename=row[0]
        value=int(row[1])
        filename = filename.split('.')[0]+'.png'
        labels[filename]=value
    return labels


train_images_path = '/home/msobroza/these/WhalesImages/data/img_train/'
test_images_path = '/home/msobroza/these/WhalesImages/data/img_sergio_test/'
train_labels_path = '/home/msobroza/these/WhalesImages/data/train.csv'
load_data = False
load_model = True
if __name__ == "__main__":
    if load_data == True:
        labels =  load_labels(train_labels_path)
        load_images(train_images_path, 100, labels)
        img_loc = load_images_test(test_images_path, 100)
        print('Finished!')
    else:
        if load_model == False:
           
            # Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
            X_train = HDF5Matrix('train.h5', 'my_data')
            y_train = HDF5Matrix('train.h5', 'my_labels')

            #y_test = HDF5Matrix('test.h5', 'my_labels')    

            # Test pretrained model
            #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model = vgg_net()
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=32, nb_epoch=5, verbose=1, shuffle='batch')
            # serialize model to JSON 
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")
        else:
            img_loc = load_images_test(test_images_path, 100)
            # Likewise for the test set
            X_test = HDF5Matrix('test.h5', 'my_data')
            # load json and create model
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("model.h5")
            print("Loaded model from disk")
 
            # evaluate loaded model on test data
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            result = model.predict(X_test, batch_size=32, verbose=1)
            print('Shape result: ')
            print(result.shape)
            for i in range(result.shape[0]):
                print('img i=%d , value: %f'%(i, result[i]))
            indexes_whales = np.argwhere(result > 0.5).reshape(-1)
            print('Shape whale: ')
            print(indexes_whales.shape)
            print(indexes_whales)
            print('Identified Whales...')
            for i in range(indexes_whales.shape[0]):
                print(img_loc[i])
