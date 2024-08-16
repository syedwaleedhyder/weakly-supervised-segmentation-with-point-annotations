import torch

class Config:
    batch_size = 4
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = "../Massachusetts Buildings Dataset/png/sliced"
    is_weakly_supervised = True
    point_label_percentage = 0.001  # For weak supervision
    use_wandb = True  # Set to True to enable wandb logging