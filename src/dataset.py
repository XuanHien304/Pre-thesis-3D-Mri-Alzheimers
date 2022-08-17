import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torchio as tio


def read_file(file_path):
    img = nib.load(file_path)
    img = img.get_fdata()
    return img

def get_train_transforms():
    return tio.Compose([   tio.Resize((110,110,110)),
                    tio.RandomAffine(scales=(0.8,1.2), degrees=(-20,20), translation=(0,0.2), center="image"),
                    tio.RandomFlip(axes=['LR']),
                    tio.ZNormalization(masking_method = tio.ZNormalization.mean),
                    tio.OneOf({
                            tio.RandomAffine(degrees = 15, translation = 15): 0.5,
                            tio.RandomFlip(): 0.3,
                            tio.RandomNoise(mean = 0.6): 0.2,
})])
    
def get_test_transforms():
    return tio.Compose([tio.Resize((110,110,110))])



class MRI_Dataset(Dataset):
    def __init__(self, folder_dir, dataframe, transforms=None):
        super().__init__()
        self.folder_dir = folder_dir
        self.image_labels = []
        self.transforms = transforms
        self.image_path = [self.folder_dir + i for i in os.listdir(self.folder_dir)]

        for i in os.listdir(self.folder_dir):
            self.image_labels.append((dataframe['Diagnosis'][dataframe['subj_id'] == i[5:15]]).to_list())

    def __getitem__(self, idx):
        img = self.image_path[idx]
        img = read_file(img)
        img = torch.tensor(img, dtype=torch.float)
        img = img.unsqueeze(dim=0)

        label = torch.FloatTensor(self.image_labels[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label

    def __len__(self):
        return len(self.image_path)
    
if __name__ == '__main__':   
    df = pd.read_csv('./prethesis/volume_score.csv')
    folder_dir = './prethesis/AD_data/train/'
    datas = MRI_Dataset(folder_dir, df, transforms=get_train_transforms())
    data_dataloader = DataLoader(dataset=datas, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    for image, label in data_dataloader:
        print(image.shape)
        print(label.shape)
        break
    

