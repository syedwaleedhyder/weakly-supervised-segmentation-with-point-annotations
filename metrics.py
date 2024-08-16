import torch

def iou(preds, labels, num_classes):
    smooth = 1e-6
    ious = []
    preds = torch.argmax(preds, dim=1)
    labels = torch.argmax(labels, dim=1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection + smooth) / (union + smooth))
    return torch.tensor(ious).nanmean()