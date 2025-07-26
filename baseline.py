# ---------------- baseline.py ----------------
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import cfg
from dataset import get_dataloaders
from timm.models import create_model
from utils import accuracy, save_model, load_model
# Proper deep copy
cfg_baseline = copy.deepcopy(cfg)
cfg_baseline.save_path = './checkpoints/baseline_vit_cifar100.pth'

# Ensure directory exists
os.makedirs(os.path.dirname(cfg_baseline.save_path), exist_ok=True)

def train_baseline():
    print(f"Using device: {cfg_baseline.device}")
    train_loader, _ = get_dataloaders(cfg_baseline)
    model = create_model(cfg_baseline.model_name, pretrained=True, num_classes=cfg_baseline.num_classes)
    model = model.to(cfg_baseline.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg_baseline.lr, weight_decay=cfg_baseline.weight_decay)

    for epoch in range(cfg_baseline.num_epochs):
        model.train()
        total_loss = 0
        top1_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}/{cfg_baseline.num_epochs}"):
            inputs, targets = inputs.to(cfg_baseline.device), targets.to(cfg_baseline.device)
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
        print(f"[Baseline] Epoch {epoch+1}: Loss={avg_loss:.4f}, Top-1 Accuracy={avg_acc:.2f}%")

    save_model(model, cfg_baseline.save_path)


def test_baseline():
    _, test_loader = get_dataloaders(cfg_baseline)
    model = create_model(cfg_baseline.model_name, pretrained=True, num_classes=cfg_baseline.num_classes)
    model = model.to(cfg_baseline.device)
    model = load_model(model, cfg_baseline.save_path)
    model.eval()

    top1_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(cfg_baseline.device), targets.to(cfg_baseline.device)
            outputs = model(inputs)
            top1 = accuracy(outputs, targets, topk=(1,))[0]
            top1_total += top1.item()

    avg_acc = top1_total / len(test_loader)
    print(f"[Baseline] Test Top-1 Accuracy: {avg_acc:.2f}%")


if __name__ == '__main__':
    train_baseline()
    test_baseline()
