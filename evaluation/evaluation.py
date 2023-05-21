import cv2
import numpy as np

def isBoundaryPixel(labels, i, j):
  if i > 0 and labels[i, j] != labels[i-1, j]:
    return True
  if i < labels.shape[0]-1 and labels[i, j] != labels[i+1, j]:
    return True
  if j > 0 and labels[i, j] != labels[i, j-1]:
    return True
  if j < labels.shape[1]-1 and labels[i, j] != labels[i, j+1]:
    return True
  return False

def computIntersectionMat(labels, gt):
  max_label = np.max(labels)
  max_gt = np.max(gt)
  inter_mat = np.zeros((max_gt+1, max_label+1))
  superpixels = np.zeros(max_label+1)
  gt_labels = np.zeros(max_gt+1)
  for i, j in np.ndindex(gt.shape):
    inter_mat[gt[i, j], labels[i,j]] += 1
    superpixels[labels[i, j]] += 1
    gt_labels[gt[i, j]] += 1
  
  return inter_mat, superpixels, gt_labels

def boundaryRecall(img, gt, dis):
  TP, FN = 0, 0
  h, w = img.shape

  for x, y in np.ndindex(img.shape):
    if isBoundaryPixel(gt, x, y):
      pos = False
      
      for i in range(max((x - dis), 0), min(x + dis, h - 1)):
        for j in range(max((y - dis), 0), min(y + dis, w - 1)):
          if(isBoundaryPixel(img, i, j)):
            pos = True
      if(pos):
        TP += 1
      else:
        FN += 1
  return 0 if TP + FN == 0 else TP / (TP+FN)
        
def underSegError(img, gt):
  intersec_mat, labels_cnt, gt_cnt = computIntersectionMat(img, gt)

  error = 0
  for i in range(intersec_mat.shape[1]):
    
    min_err = np.inf
    for j in range(intersec_mat.shape[0]):
      diff = labels_cnt[i] - intersec_mat[j, i]
      min_err = min(diff, min_err)
    error += min_err

  return error / (gt.shape[0] * gt.shape[1])
