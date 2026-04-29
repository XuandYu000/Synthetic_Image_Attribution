import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, scheduler, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainer_init(args)

    def trainer_init(self, args):
        self.device = args.device
        if args.save_dir is not None:
            self.save_dir = os.path.join(args.save_dir, args.model_name)
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            sample_data, sample_label = batch
            sample_data = sample_data.to(self.device)
            sample_label = sample_label.to(self.device)
            output = self.model(sample_data)
            loss = self.criterion(output, sample_label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.train_loader)
        return train_loss

    def test(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for batch in self.test_loader:
                sample_data, sample_label = batch
                sample_data = sample_data.to(self.device)
                sample_label = sample_label.to(self.device)
                output = self.model(sample_data)
                loss = self.criterion(output, sample_label)
                batch_size = sample_label.size(0)
                val_loss += loss.item() * batch_size
                correct += (output.argmax(dim=1) == sample_label).sum().item()
                total += batch_size

        if total > 0:
            val_loss /= total
            val_acc = correct / total
        else:
            val_loss = 0.0
            val_acc = 0.0
        return val_loss, val_acc    

    def train(self, num_epochs):
        best_val_loss = float('inf')
        for epoch in tqdm(range(num_epochs), desc="Training"):
            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.test()
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}")
            print(f"Current LR: {current_lr:.8f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"best_model.pth"))
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)

        return output