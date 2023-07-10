import os.path
from cProfile import label
import json
from math import ceil
import scipy.misc
from skimage import transform
import random
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path:str, label_path:str, batch_size:int, image_size:list, rotation:bool= False, mirroring:bool= False, shuffle:bool= False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path=file_path
        self.label_path=label_path
        self.batch_size=batch_size
        self.image_size=image_size
        self.rotation=rotation
        self.mirroring=mirroring
        self.shuffle=shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.batch_index=0
        self.current_epoch_num=0
        with open(self.label_path,'r') as i:
            self.labels:dict=json.load(i)
        self.image_keys=list(self.labels.keys())
        if self.shuffle:
            random.shuffle(self.image_keys)
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = []
        labels = []
        start=self.batch_index*self.batch_size
        end=start+self.batch_size
        if end<=len(self.labels):     #since JSON file size equals image dataset size
            for i in range(start,end):
                key_value=self.labels[self.image_keys[i]]
                img_from_dataset=np.load(self.file_path+str(self.image_keys[i])+'.npy')
                img_from_dataset = transform.resize(img_from_dataset,self.image_size)
                if self.rotation or self.mirroring:
                    img_from_dataset=self.augment(img_from_dataset)
                images.append(img_from_dataset)
                labels.append(key_value)
            self.batch_index+=1
            if self.shuffle:
                random.shuffle(self.image_keys)
            if len(labels) == len(self.labels):
                self.current_epoch_num+=1
                if self.shuffle:                #shuffle after every epoch
                    random.shuffle(self.image_keys)
                    # paired = list(zip(images, labels))
                    # random.shuffle(paired)
                    # images, labels = zip(*paired)
        else:                                               #to fill in shortage of image data in the last batch
            excess=len(self.labels)-start
            for j in range(start,len(self.labels)):
                key_value=self.labels[self.image_keys[j]]
                img_from_dataset = np.load(self.file_path + str(self.image_keys[j]) + '.npy')
                img_from_dataset = transform.resize(img_from_dataset,self.image_size)
                if self.rotation or self.mirroring:
                    img_from_dataset=self.augment(img_from_dataset)
                images.append(img_from_dataset)
                labels.append(key_value)
            for k in range(0,(self.batch_size-excess)):
                key_value = self.labels[self.image_keys[k]]
                img_from_dataset = np.load(self.file_path + str(self.image_keys[k]) + '.npy')
                img_from_dataset = transform.resize(img_from_dataset,self.image_size)
                if self.rotation or self.mirroring:
                    img_from_dataset=self.augment(img_from_dataset)
                images.append(img_from_dataset)
                labels.append(key_value)
            self.batch_index=0
            self.current_epoch_num+=1
            if self.shuffle:
                random.shuffle(self.image_keys)
                # paired=list(zip(images,labels))
                # random.shuffle(paired)
                # images,labels=zip(*paired)
        images_array=np.array(images)
        labels_array=np.array(labels)
        return images_array,labels_array
        #return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring and random.random()<0.5:
            img=np.fliplr(img)
        if self.rotation and random.random()<0.5:
            a=random.choice([90,180,270])
            img=np.rot90(img, k=a // 90)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch_num

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images,labels=self.next()
        _,axs=plt.subplots(ceil((len(images)/3)),3,sharex=True,sharey=True)
        axs=axs.flatten()
        for i,j,ax in zip(images,labels,axs):
            ax.title.set_text(self.class_name(j))
            ax.imshow(i)
        plt.show()