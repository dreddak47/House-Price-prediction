import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch
from PIL import Image




class customdata(Dataset):
    PATH='GVSS_Vision_Data/Annotation.xlsx'
    def __init__(self,root,train=True,transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms

        image1 = "images/"
        self.image1=image1
        self.image1_folder = sorted([os.path.join(self.root+image1,x) for x in os.listdir(self.root+image1)])
    
    

    def __len__(self):
        return len(self.image1_folder)
    
    def read_files(self,filename):
        P = sorted(os.listdir(filename))
        P = [i for i in P if i[-1]=='g']
        return P


    def __getitem__(self, index):

        images_name = self.read_files(self.root+self.image1)
        imgn=images_name[index]
        i=imgn.split('_')[0]
        text=[]
        df=pd.read_excel(self.PATH)
        txt=df.loc[df['Index']==i]
        text.append(txt['n_Area'].to_numpy()[0])
        text.append(txt['Area per room'].to_numpy[0])
        text.append(txt['No_of_Bedrooms'].to_numpy()[0])
        text.append(txt['No_of_Bathrooms'].to_numpy()[0])
        text.append(txt['Total room'].to_numpy()[0])
        text.append(txt['If1'].to_numpy()[0])
        text.append(txt['If2'].to_numpy()[0])
        text.append(txt['n_Price'].to_numpy()[0])
        img1 =  cv2.imread(self.root+self.image1+images_name[index])
        img1 = cv2.resize(img1,(img1.shape[0]*2,img1.shape[1]*2))
        img1=np.array(img1)
        
        
        if self.transforms is not None:
            img1 = self.transforms(img1)
        return (img1,text)


class customdata1(Dataset):
    PATH='GVSS_Vision_Data/Annotation.xlsx'
    def __init__(self,root,train=True,transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms

        image1 = "images/"
        self.image1=image1
        self.image1_folder = sorted([os.path.join(self.root+image1,x) for x in os.listdir(self.root+image1)])
    

    def __len__(self):
        return len(self.image1_folder)
    def read_files(self,filename):
        P = sorted(os.listdir(filename))
        P = [i for i in P if i[-1]=='g']
        return P


    def __getitem__(self, index):

        images_name = self.read_files(self.root+self.image1)
        imgn=images_name[index]
        i=int(imgn.split('_')[-2])
        text=[]
        df=pd.read_excel(self.PATH)
        txt=df.loc[df['Index']==i]
        text.append(txt['n_Area'].to_numpy()[0])
        text.append(txt['Area per room'].to_numpy()[0])
        text.append(txt['No_of_Bedrooms'].to_numpy()[0])
        text.append(txt['No_of_Bathrooms'].to_numpy()[0])
        text.append(txt['Total room'].to_numpy()[0])
        text.append(txt['If1'].to_numpy()[0])
        text.append(txt['If2'].to_numpy()[0])
        text.append(txt['n_Price'].to_numpy()[0])
        img1 =  cv2.imread(self.root+self.image1+images_name[index])
        img1 = cv2.resize(img1,(img1.shape[0]*2,img1.shape[1]*2))
        img1=np.array(img1)
        
        if self.transforms is not None:
            img1=Image.fromarray(img1)
            img1 = self.transforms(img1)
        return (img1,np.array(text).astype(np.float32))


# mt = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Resize((256,256))
#     ]
# )

# dataset = customdata(root="Data/train/",train=True,transforms=mt)

# loader = DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True)

# for i,j in loader:
#     print(i.shape,j.shape)

