import cv2
import numpy as np
from evaluation import boundaryRecall
seg = "38082.seg"

def SLIC(img, max_iter=10):
    #設定SLIC初始化設定
    slic = cv2.ximgproc.createSuperpixelSLIC(img) 
    slic.iterate(max_iter)
    mask_slic = slic.getLabelContourMask() #建立超像素的遮罩，mask_slic數值為1
    label_slic = slic.getLabels()        #獲得超像素的標籤
    # number_slic = slic.getNumberOfSuperpixels()  #獲得超項素的數量
    mask_inv_slic = cv2.bitwise_not(mask_slic)  
    img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic) #在原圖中繪製超像素邊界
    # cv2.imwrite('./SLIC.png', img_slic)
    return label_slic #將繪製邊界的圖片儲存



def gt_berkely_labels(gt_file):
  with open(gt_file, "r") as f:
    seg_label = f.readlines()
  label_list = []
  height = int(seg_label[5].split(' ')[1])
  width = int(seg_label[4].split(' ')[1])
  for lines in seg_label[11:]:
    [obj, x, y1, y2] = lines.split(' ')
    label_list.append([int(obj), int(x), int(y1), int(y2)])

  gt = np.full((height, width), -1)

  for [obj, x, y1, y2] in label_list:
    gt[x, y1:y2+1] = obj
  return gt

