#======Description=====
#Author: Daniel Beeston
#Purpose: Implementation of a custom Pytorch dataset for wildfire smoke videos

import os
import torch
from torchvision import transforms
import csv
import numpy as np
import cv2
from PIL import Image


class SmokeVideoDataLoader(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,    #path to database root
                 label_file: str,   #path to label file
                 video_directory:str,  #path to video directory
                 frameskip: int,    #number of frames skipped between frame capture
                 num_images: int,   #maximum number of images captured
                 start_frame: int,   #start frame of image capture
                 batch_size: int    #number of videos loaded per batch
                 ):
        self.root_path = root_path
        self.label_file = label_file
        self.video_directory = video_directory
        self.batch_size = batch_size
        self.num_images = num_images
        self.frameskip = frameskip
        self.start_frame = start_frame

        self.video_list = os.listdir(self.root_path+'/'+self.video_directory)
        self.video_list.sort()
        self.current_batch = 0
        self.batch_index = 0

        self.label_list =[]

        with open(root_path+'/'+label_file, newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                self.label_list.append(int(row[0]))

    def __len__(self):
        return len(self.video_list)

    def __getPILlist__(self,index):
        videopath = self.root_path+'/'+self.video_directory+'/'+self.video_list[index]
        imagelist = []

        cap = cv2.VideoCapture(videopath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #CV_CAP_PROP_POS_FRAMES
        current_images = 0

        i=self.start_frame
        while(cap.isOpened()):
            cap.set(1,i)
            ret, frame = cap.read()

            if i>total_frames or current_images>self.num_images:
                break

            img = frame
            img = img[180:180+1080]    #crop top of image
            img = img[0:850]          #crop bottom of image
            img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)  #downsize image to 40%
            #img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX)

            color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image=Image.fromarray(color_coverted)
            imagelist.append(pil_image)

            current_images += 1

            i += self.frameskip
        cap.release()

        return (imagelist)

    def __getlabel__(self,index):
        return self.label_list[index]

    def __get__(self,index):
        item = self.__getPILlist__(index)
        return torch.stack([transforms.functional.to_tensor(img) for img in item])

    def __getitem__(self, index):
        return self.__get__(index),self.__getlabel__(index)

    def __getbatch__(self):
        batch_index = self.batch_index
        batch_size = self.batch_size
        tensorlist = []
        labellist = []

        if batch_index+batch_size > len(self.video_list): #return if unable to construct full batch
            return False,labellist

        for x in range(batch_index,batch_index+batch_size):
            tensorlist.append(self.__get__(x))
            labellist.append(self.__getlabel__(x))

        self.batch_index+=self.batch_size
        self.current_batch+=1

        return torch.stack(tensorlist), labellist






