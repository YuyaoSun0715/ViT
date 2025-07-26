class Config:
    # Dataset
    dataset = 'CIFAR100'
    data_dir = './data'
    num_classes = 100

    # Model
    model_name = 'vit_base_patch16_224'
    prompt_length = 5  # number of prompt tokens

    # Training
    batch_size = 64
    lr = 5e-4
    weight_decay = 0.05
    num_epochs = 30
    device = 'cuda'  # or 'cpu'

    # Save path
    save_path = './checkpoints/vpt_vit_cifar100.pth'

cfg = Config()
