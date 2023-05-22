import cv2
import numpy as np
import sys
import os
import time
from readData import gt_berkely_labels, SLIC
from evaluation import boundaryRecall, underSegError

import slic 

if __name__ == "__main__":
  
  dataset = sys.argv[1]
  
  br_list = []
  time_list = []
  useg_err_list = []
  if dataset == "berkely":
    print("Processing Berkely Dataset...")
    data_dir = sys.argv[2]
    seg_dir = sys.argv[3]
      
    for img_file in os.listdir(data_dir):
      seg_file = os.path.join(seg_dir, img_file[:-3] + 'seg')

      gt_label_mat = gt_berkely_labels(seg_file)
      img = cv2.imread(os.path.join(data_dir, img_file))
      
      start_time = time.time()

      # labels = SLIC(img)
      p = slic.SLIC(img, k=4096, m=40, max_iter=10)
      p.init_clusters()
      p.move_clusters()
      p.repeat()
      labels = p.label

      time_list.append(time.time() - start_time)
      
      br = boundaryRecall(labels, gt_label_mat, 2)
      br_list.append(br)
      us_err = underSegError(labels, gt_label_mat)
      useg_err_list.append(us_err)

  print('================== evaluation result ==================')
  print('k=4096, m=40, max_iter=10')
  print('DataSet:                     |', sys.argv[1])
  print('Total Images                 |', len(br_list))
  print('Avg Boundary Recall          |', sum(br_list)/len(br_list))    
  print('Avg Under Segmentation Error |', sum(useg_err_list)/len(useg_err_list))    
  print('Avg Processing Time          |', sum(time_list)/len(time_list))  
  print('=======================================================')  
