import torch
import math
import numpy as np

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.avg * (self.count / (self.count + n)) + val * (n / (self.count + n))
        self.count += n

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def get_f_score(dist_label, dist_pred, threshold):
    batch_size = dist_label.shape[0]
    num_threshold = len(threshold)
    num_label = dist_label.shape[1]
    num_predict = dist_pred.shape[1]

    num_label = torch.tensor([num_label]*batch_size).to(dist_pred.device)
    num_predict = torch.tensor([num_predict]*batch_size).to(dist_pred.device)
    f_scores = torch.zeros((batch_size, num_threshold)).to(dist_pred.device)
    
    for i in range(num_threshold):
        num = torch.where(dist_label <= threshold[i], 1, 0).sum(1)
        
        recall = 100.0 * num / num_label
        num = torch.where(dist_pred <= threshold[i], 1, 0).sum(1)
        
        precision = 100.0 * num / num_predict
		# f_scores.append((2*precision*recall)/(precision+recall+1e-8))
        f_scores[:, i] = (2*precision*recall)/(precision+recall+1e-8)
    return f_scores