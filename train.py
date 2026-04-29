import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset
from model import get_model
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--train_csv", type=str, default="dataset/Data/Data/training.csv")
    parser.add_argument("--data_root", type=str, default="dataset/Data/")
    parser.add_argument("--input_size", type=tuple, default=(224, 224))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=100)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = get_model(args.model_name, args.num_classes, args.pretrained)
    args.input_size = model.pretrained_cfg['input_size'][1:] # remove the batch dimension
    print(args)

    dataset = MyDataset(args.train_csv, args.data_root, args.input_size, mode="train")
    test_size = args.test_size
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_lr
    )

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, scheduler, args)
    trainer.train(args.num_epochs)