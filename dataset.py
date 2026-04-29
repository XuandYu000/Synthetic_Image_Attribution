import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvtf

class MyDataset(Dataset):
    def __init__(self, csv_path="dataset/Data/Data/training.csv", data_root="dataset/Data/", target_size=(224, 224), mode="train"):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.mode = mode
        
        self.default_transform = tvtf.Compose([
            tvtf.Resize(target_size),
            tvtf.ToTensor(),
            tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # only for train
        self.transform = tvtf.Compose([
            tvtf.GaussianBlur(kernel_size=5),
            tvtf.RandomAutocontrast(p=0.25),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[1]
        img_path = os.path.join(self.data_root, img_path)
        img = Image.open(img_path).convert("RGB")

        # train, test, use "y" as label
        if self.mode == "train":
            img = self.transform(img)
            label = int(row[2])
        # val, predict, use "ID" as label for consistency
        else:
            label = row[0]

        # default transform
        img = self.default_transform(img)
        
        return img, label

if __name__ == "__main__":
    dataset = MyDataset()
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for batch in dataloader:
        sample_data, sample_label = batch
        print(sample_data.shape)
        print(sample_label)
        break