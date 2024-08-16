import matplotlib.pyplot as plt
import torch
import cv2
import os
from datetime import datetime

def visualize_results(model, test_dataset, device, is_weakly_supervised, num_samples=5):
    model.eval()
    
    # Create a directory for the output files
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f'visualization_{"weakly" if is_weakly_supervised else "fully"}_{timestamp}'
    os.makedirs(folder_name, exist_ok=True)
    
    for i in range(num_samples):
        if is_weakly_supervised:
            image, mask_one_hot, point_mask = test_dataset[i]
        else:
            image, mask_one_hot = test_dataset[i]

        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)

        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        mask_indices = torch.argmax(mask_one_hot, dim=0).squeeze().cpu().numpy()

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        image_renormalized = image.squeeze() * std[:, None, None] + mean[:, None, None]
        image_rgb = cv2.cvtColor(image_renormalized.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)

        # Save the images individually
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.savefig(os.path.join(folder_name, f'original_image{i+1}.png'), bbox_inches='tight', pad_inches=0)
        
        plt.imshow(mask_indices, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(folder_name, f'gt{i+1}.png'), bbox_inches='tight', pad_inches=0)
        
        plt.imshow(pred, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(folder_name, f'pred{i+1}.png'), bbox_inches='tight', pad_inches=0)

        if is_weakly_supervised:
            point_mask_display = point_mask.squeeze().cpu().numpy()
            plt.imshow(point_mask_display, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(folder_name, f'point_mask{i+1}.png'), bbox_inches='tight', pad_inches=0)
    
    print(f'Files saved in directory: {folder_name}')

