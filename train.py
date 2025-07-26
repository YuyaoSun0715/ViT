import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import cfg
from dataset import get_dataloaders
from model import VPTViT
from utils import accuracy, save_model

def train():
    train_loader, test_loader = get_dataloaders(cfg)
    model = VPTViT(cfg).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        top1_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            top1 = accuracy(outputs, targets, topk=(1,))[0]
            top1_total += top1.item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = top1_total / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Top-1 Accuracy={avg_acc:.2f}%")

    save_model(model, cfg.save_path)

if __name__ == '__main__':
    train()