import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class Archi(nn.Module):
    def __init__(self):
        super().__init__()


        self.backbone1 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

        self.block3_1 = nn.Conv2d(512,100,3,1,1)
        self.block3_2 = nn.Conv2d(100,20,2,2,1)


        self.fc = nn.Sequential(
            nn.Linear(500,100),
            nn.GELU(),
            nn.Linear(100,60),
            nn.GELU(),
            nn.Linear(60,30),
            nn.GELU(),
            nn.Linear(30,10),
            nn.GELU(),
            nn.Linear(10,1),
        )

        # self.fc3 = nn.Sequential(nn.Linear(2,2),nn.Softmax(dim=-1))

    def forward(self,x):
        x =  self.backbone1(x)
        x = F.adaptive_avg_pool2d(x,(8,8))
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = x.reshape(x.shape[0],-1)
        
        x = self.fc(x)
        return x

class Archi2(nn.Module):
    def __init__(self):
        super().__init__()


        self.backbone1 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

        self.block3_1 = nn.Conv2d(512,100,3,1,1)
        self.block3_2 = nn.Conv2d(100,20,2,2,1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.fc = nn.Sequential(
            nn.Linear(500,256),
            nn.GELU(),
            nn.Linear(256,64),
            nn.GELU(),
            nn.Linear(64,16),
            nn.GELU(),
            nn.Linear(16,10),
        )

        # self.fc3 = nn.Sequential(nn.Linear(2,2),nn.Softmax(dim=-1))

    def forward(self,x):
        x =  self.backbone1(x)
        x = F.adaptive_avg_pool2d(x,(8,8))
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = x.reshape(x.shape[0],-1)
        
        x = self.fc(x)
        return x
    
from transformers import ViTModel,ViTFeatureExtractor

class Archi3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit=ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size,32),
            nn.GELU(),
            nn.Linear(32,1),
        )

        # self.fc3 = nn.Sequential(nn.Linear(2,2),nn.Softmax(dim=-1))
    def forward(self,x):
        x =  self.vit(x)
        x = x.pooler_output
        
        x = self.fc(x)
        return x 

class combined(nn.Module):
    def __init__(self, archi_model, text_feature_dim):
        super().__init__()
        self.archi_model = archi_model  # Your pretrained Archi model
        self.text_fc = nn.Sequential(
            nn.Linear(text_feature_dim, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8,4)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(14, 8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
            nn.Linear(4, 1),
        )

    def forward(self, image, text):
        image_features = self.archi_model(image)
        text_features = self.text_fc(text)
        combined_features = torch.cat((image_features, text_features), dim=1)
        price = self.combined_fc(combined_features)
        return price
    
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# def test():
#     model = Archi3().cpu()
#     inp1 = np.zeros((1,224,224))
#     #inp2 = torch.randn((1,3,128,128)).cuda()
#     #inp3 = torch.randn((1,19)).cuda()
#     inputs = feature_extractor(images=inp1, return_tensors="pt")
#     out = model(**inputs)
#     print(out.shape)
#     print(out)


# test()
