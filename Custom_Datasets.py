import os
import torch
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from PIL import Image, ImageFilter, ImageStat
from torchvision import transforms




class Custom_Datasets(Dataset): 
    def __init__(self,root, datatxt, transform=None): 
        self.root= root
        self.transform = transform
        
        fh = open(datatxt, 'r',encoding="utf_8_sig") 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
        self.imgs = imgs

    def __getitem__(self, index):
        
            fn, label = self.imgs[index] 
            img = Image.open(os.path.join(self.root,fn))
            
            if self.transform is not None:
                img = self.transform(img) 
                
            return img,label #輸出是tensor型態

    def __len__(self): 
        return len(self.imgs)



if __name__ == "__main__":
    
    
    
    
    #trans = transforms.Compose([ transforms.Resize((32,224)),transforms.ToTensor()])
    trans = transforms.Compose([ transforms.Resize((224,224)),transforms.ToTensor()])
    tensor_to_Pil = transforms.ToPILImage()
    
    train_data = Custom_Datasets(root='./datasets/PetImages/PetImages',datatxt='./datasets/PetImages/train.txt',transform=trans) 
    #train_data = Custom_Datasets(root='./datasets/text/HR',datatxt='./datasets/text/train.txt',transform=trans)
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True,num_workers=int(0))
    for step,(img,label) in enumerate(train_dataloader):

        for i in img:
            plt.imshow(tensor_to_Pil(i))
            plt.show()
        #print(label)

