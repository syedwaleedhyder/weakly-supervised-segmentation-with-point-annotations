import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from datetime import datetime
import wandb

from config import Config
from dataset import SegmentationDataset
from model import UNet
from loss import PartialCrossEntropyLoss
from metrics import iou
from visualization import visualize_results

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes, is_weakly_supervised, use_wandb):
    model.train()
    total_loss = 0
    total_iou = 0.0
    count = 0
    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        images, masks = batch[0], batch[1]
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        if is_weakly_supervised:
            mask_points = batch[2].to(device)
            loss = criterion(outputs, masks, mask_points)
        else:
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iou_score = iou(outputs, masks, num_classes)
        total_iou += iou_score.item()
        count += 1
        avg_loss = total_loss / count
        avg_iou = total_iou / count
        progress.set_postfix({'loss': avg_loss, 'iou': avg_iou})
        
        if use_wandb:
            wandb.log({"batch_loss": loss.item(), "batch_iou": iou_score.item()})
    
    return avg_loss, avg_iou

def evaluate(model, dataloader, criterion, device, num_classes, is_weakly_supervised):
    model.eval()
    total_loss = 0
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            images, masks = batch[0], batch[1]
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            if is_weakly_supervised:
                mask_points = batch[2].to(device)
                loss = criterion(outputs, masks, mask_points)
            else:
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            iou_score = iou(outputs, masks, num_classes)
            total_iou += iou_score.item()
            count += 1
    avg_loss = total_loss / count
    avg_iou = total_iou / count
    return avg_loss, avg_iou

def log_results(log_file, epoch, train_loss, train_iou, val_loss, val_iou):
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Epoch', 'Train Loss', 'Train IoU', 'Val Loss', 'Val IoU'])
    
    with open(log_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([epoch, train_loss, train_iou, val_loss, val_iou])

def main():
    config = Config()
    
    if config.use_wandb:
        wandb.init(project="building-segmentation", config=vars(config))
    
    train_image_dir, train_mask_dir = os.path.join(config.base_dir, 'train'), os.path.join(config.base_dir, 'train_labels')
    val_image_dir, val_mask_dir = os.path.join(config.base_dir, 'val'), os.path.join(config.base_dir, 'val_labels')
    test_image_dir, test_mask_dir = os.path.join(config.base_dir, 'test'), os.path.join(config.base_dir, 'test_labels')
    class_dict_path = os.path.join(config.base_dir, 'label_class_dict.csv')

    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, class_dict_path, point_label=config.is_weakly_supervised, point_label_percentage=config.point_label_percentage)
    val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, class_dict_path, point_label=config.is_weakly_supervised, point_label_percentage=config.point_label_percentage)
    test_dataset = SegmentationDataset(test_image_dir, test_mask_dir, class_dict_path, point_label=config.is_weakly_supervised, point_label_percentage=config.point_label_percentage)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    num_classes = len(train_dataset.class_dict)
    print(f'{num_classes=}')

    model = UNet(n_channels=3, n_classes=num_classes).to(config.device)
    criterion = PartialCrossEntropyLoss() if config.is_weakly_supervised else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    if config.use_wandb:
        wandb.watch(model, log="all")
    
    # Create a log file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'training_log_{timestamp}.csv'
    
    best_val_iou = 0
    for epoch in range(config.num_epochs):
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, config.device, num_classes, config.is_weakly_supervised, config.use_wandb)
        val_loss, val_iou = evaluate(model, val_loader, criterion, config.device, num_classes, config.is_weakly_supervised)
        print(f'Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')
        
        # Log the results
        log_results(log_file, epoch+1, train_loss, train_iou, val_loss, val_iou)
        
        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou
            })

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = 'model_best_buildings.pth'
            torch.save(model.state_dict(), best_model_path)
            if config.use_wandb:
                wandb.save(best_model_path)
    
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_iou = evaluate(model, test_loader, criterion, config.device, num_classes, config.is_weakly_supervised)
    print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}')
    
    # Log the final test results
    with open(log_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Final Test Results', '', '', 'Test Loss', 'Test IoU'])
        csv_writer.writerow(['', '', '', test_loss, test_iou])
    
    if config.use_wandb:
        wandb.log({"test_loss": test_loss, "test_iou": test_iou})
        wandb.finish()

    visualize_results(model, test_dataset, config.device, config.is_weakly_supervised)

if __name__ == '__main__':
    main()