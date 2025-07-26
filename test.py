import torch
from config import cfg
from dataset import get_dataloaders
from model import VPTViT
from utils import load_model, accuracy

def test():
    _, test_loader = get_dataloaders(cfg)
    model = VPTViT(cfg).to(cfg.device)
    model = load_model(model, cfg.save_path)
    model.eval()

    top1_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            outputs = model(inputs)
            top1 = accuracy(outputs, targets, topk=(1,))[0]
            top1_total += top1.item()

    avg_acc = top1_total / len(test_loader)
    print(f"Test Top-1 Accuracy: {avg_acc:.2f}%")

if __name__ == '__main__':
    test()
