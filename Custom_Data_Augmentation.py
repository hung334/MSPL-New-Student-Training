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
import random
from random import choice,shuffle
from PIL import Image

class Aug:
    def __init__(self):
        pass

    def __call__(self, img):
        
        test_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)#PIL -> CV 
        canny_img = self.canny(test_img.copy())
        noise_img = self.add_noise(canny_img.copy())
        noise_img = Image.fromarray(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB))#轉換回來成PIL
                        
        return noise_img
    
    def add_noise(self,image, mode = 'all'):
        h, w = image.shape
        if mode == 'all' or mode == 'salt':
            times = random.randint(800, 1000)
            for i in range(times):
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)
    
                image[y][x] = 255
                
        if mode == 'all' or mode == 'pepper':
            for i in range(5000):
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)
    
                image[y][x] = 0
            
        return image
    
    def canny(self,image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(image = gray, threshold1 = 60, threshold2 = 200)
        return canny









if __name__ == "__main__":
    
    
    
    test_img = Image.open('./datasets/cat.jpg')
    
    trans = transforms.Compose([ transforms.Resize((224,224)),
                                transforms.RandomRotation((90), expand=True), #隨機旋轉
                                transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),#亮度、對比、飽和、色相
                                ])
    
    aug_img = trans(test_img)
    
    aug2_img = Aug()(test_img)
    
    trans2 = transforms.Compose([
                                
                                transforms.Resize((224,224)),
                                transforms.RandomRotation((90), expand=True), #隨機旋轉
                                Aug(),
                                ])
    
        
    aug3_img = trans2(test_img)

    
    trans3 = transforms.Compose([
                                
                                transforms.Resize((224,224)),
                                Aug(),
                                transforms.ToTensor(),
                                ])

    tensor_to_Pil = transforms.ToPILImage()
    
    
    from Custom_Datasets import Custom_Datasets
    
    train_data = Custom_Datasets(root='./datasets/PetImages/PetImages',datatxt='./datasets/PetImages/train.txt',transform=trans3) 
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True,num_workers=int(0))
    for step,(img,label) in enumerate(train_dataloader):

        for i in img:
            plt.imshow(tensor_to_Pil(i))
            plt.show()
    


