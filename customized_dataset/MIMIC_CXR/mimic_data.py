
from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle
#import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn   
from torchvision import transforms, datasets


import csv
from PIL import Image
from PIL import ImageFile
import numpy as np
import pandas as pd
from torchmetrics import AUROC
import torch.nn as nn
import torch.nn.functional as F
from customized_dataset.MIMIC_CXR.util import TwoCropTransform
from torch.utils.data.sampler import SubsetRandomSampler


    
class MimicCxrDataset():
    def __init__(self,y,csv_file,transform):
        super(MimicCxrDataset,self).__init__()
        self.img_paths=[]
        self.transform = transform
        self.chexpert_labels = []
        self.label_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        self.base_path = "/vol/space/projects/MIMIC-CXR/mimic-cxr_ap-pa_dataset/images/"
        self.y = y

        #dict=pickle.load(open("idx2patient_425.pickle","rb"))
        reports=pickle.load(open("/u/home/mohammed/Graphormer/pickles/Frontal_Graph_3000.pkl","rb"))
        df=pd.read_csv(csv_file)

        for idx in range(len(y)):
            
            patient = reports[y[idx].item()]
            x=patient.split("/")
            patient_id = x[1]
            study_id = x[2].split(".")[0]
            #pid=dict[y[idx].item()][0]
            #sid=dict[y[idx].item()][1]
            pid = int(patient_id[1:])
            sid = int(study_id[1:])
            #print(pid,sid,type(pid))
            row = (df[(df.patient_id == pid)  & (df.study_id == sid)])
            if not row.empty:
 
                item=list(row["img_file"])[0]
                print(item)
                self.img_paths.append(os.path.join(self.base_path,item))
                #self.chexpert_labels.append(tuple(row[label_name] for label_name in self.label_names))
            else:
                print("MISSING DATA")
                print(pid,sid)

    def __getitem__(self, item):

        image_path=self.img_paths[item]
        img=Image.open(image_path).convert("RGB")



        img=self.transform(img)
        #labels = self.chexpert_labels[item]

        l=[]
        #for i,label in enumerate(labels):
         #   l.append(torch.tensor(int(label)))
        #l=torch.tensor(l,dtype=torch.float32)
        return img
    def __len__(self):
        return len(self.img_paths)

class create_loader():
    def __init__(self):
        super(create_loader,self).__init__()
    
        self.t_mean = [0.47472174332872247]
        self.t_std =  [0.30418304309833266]
        #self.v_mean = [0.4768379613202303]
        #self.v_std =  [0.30315016207424433]
        self.train_csv = "/vol/space/projects/MIMIC-CXR/mimic-cxr_ap-pa_dataset/combined_frontal_list.csv"
        self.batch_size = 64
        self.size = 224
        self.num_workers = 0
        self.train_size = 0.8
        #self.val_size = 0

        self.t_normalize = transforms.Normalize(mean=self.t_mean, std=self.t_std)

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.size, scale=(0.8, 1.)),  # opt.size = 224 #20 percent to 100 percent
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4,0.4,0,0)
            ], p=0.8),
            transforms.ToTensor(),
            self.t_normalize,
            ])



    def bring_loaders(self,y):

    
        batch_dataset = MimicCxrDataset(y,self.train_csv,transform =self.train_transform)
        loader = torch.utils.data.DataLoader(batch_dataset,batch_size=64)

        #print(batch_dataset.shape)
        batch = next(iter(loader))
        print("batch size = ",batch.shape)
        #print((batch_dataset[0][0].shape))
        #print(len(batch_dataset[0]))       
       

        return batch