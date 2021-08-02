import numpy as np
import torch


def IoU(target, output):
    intersection_all = np.sum(np.logical_and(target, output))
    union_all = np.sum(np.logical_or(target, output))
    if union_all == 0:
        iou = 1
    else:
        iou = intersection_all/union_all
    return iou

def BER(target, output):
    pass

def MAE(target, output):
    # predict = (torch.argmax(output, 1)).float()
    # target = target.float()
    mae = np.mean(np.absolute((output - target)))
    return mae

def PicAcc(target, output):
    # predict = torch.argmax(output.long(), 1) + 1
    # target = target.long() + 1
    acc = np.sum(np.logical_not(np.logical_xor(target, output)))

    # pixel_labeled = torch.sum(target > 1)
    # pixel_correct = torch.sum((predict == target) * (target > 1))
    # assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    # return pixel_correct, pixel_labeled
    return acc