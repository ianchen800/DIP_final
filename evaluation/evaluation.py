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
        