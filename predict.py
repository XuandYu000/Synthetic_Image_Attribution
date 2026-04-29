import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import get_model
from trainer import Trainer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="results/resnet50/best_model.pth")
    parser.add_argument("--test_csv", type=str, default="dataset/Data/Data/test.csv")
    parser.add_argument("--data_root", type=str, default="dataset/Data/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default="./submission.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    model = get_model(args.model_name, args.num_classes, args.pretrained)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    input_size = model.pretrained_cfg['input_size'][1:] # remove the batch dimension
    dataset = MyDataset(args.test_csv, args.data_root, input_size, mode="predict")
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    trainer = Trainer(model, None, test_loader, None, None, None, args)
    all_ids = []
    all_targets = []
    for batch in tqdm(test_loader, desc="Predicting"):
        sample_data, sample_label = batch
        sample_data = sample_data.to(device)
        logits = trainer.predict(sample_data)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        if torch.is_tensor(sample_label):
            sample_ids = sample_label.cpu().tolist()
        else:
            sample_ids = list(sample_label)

        all_ids.extend(sample_ids)
        all_targets.extend(preds)
    
    submission = pd.DataFrame({"ID": all_ids, "Target": all_targets})
    submission["ID"] = submission["ID"].astype(int)
    submission["Target"] = submission["Target"].astype(int)
    submission.to_csv(args.output_csv, index=False)
    print(f"Saved prediction to {args.output_csv}")

if __name__ == "__main__":
    main()
