from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(cfg):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = datasets.CIFAR100(root=cfg.data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root=cfg.data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader