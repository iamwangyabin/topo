import os
from PIL import Image
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

def default_loader(path):
    return Image.open(path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((60, 180)),
                                            torchvision.transforms.ToTensor(),
                                            #torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                            ])

#path='/home/wang/桌面/top/testData/'
path='/home/wang/桌面/top/testData2/'

class myDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, loader=default_loader):
        img_list = []
        for i in os.listdir(path):
            img_list.append(path + i)
        self.root = path
        self.transform = transform
        self.len = len(os.listdir(path))
        self.img_list = img_list
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = self.loader(img_path).convert('L')
        img = img.crop((81, 161, 575, 324))
        img = self.transform(img)
        name = img_path.split('/')[-1]
        volfrac = name.split('_')[0]
        penal = name.split('_')[1]
        rmin = name.split('_')[2][:-4]
        return img,img_path,volfrac,penal,rmin

    def __len__(self):
        return self.len


dataset = myDataset(path, transform, default_loader)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=3)


def show():
    to_pil_image = torchvision.transforms.ToPILImage()
    cnt = 0
    for image, label, k, j, l in data_loader:
        if cnt>=1:      # 只显示3张图片
            break
        print(label)    # 显示label
        img = to_pil_image(image[0])
        plt.imshow(img)
        cnt += 1

for image, label, k, j, l in data_loader:
    if cnt>=1:      # 只显示3张图片
        break
    print(label)    # 显示label
    print(model(image))
    cnt += 1