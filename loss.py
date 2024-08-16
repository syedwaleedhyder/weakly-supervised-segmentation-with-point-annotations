import torch
import torch.nn as nn

class PartialCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PartialCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask):
        mask = mask.expand_as(targets)
        inputs_masked = inputs * mask
        targets_masked = targets * mask
        log_softmax = torch.nn.functional.log_softmax(inputs_masked, dim=1)
        loss = -torch.sum(targets_masked * log_softmax, dim=1)
        loss_masked = loss * mask[:, 0, :, :]
        return loss_masked.sum() / mask.sum()