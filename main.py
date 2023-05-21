import math
import numpy as np
from tqdm import trange
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
# import torch

# device = torch.device('mps')
# print(device)

def Show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyWindow(name)

class Cluster(object):
    def __init__(self, l, a, b, i, j):
        self.update(l, a, b, i, j)
        # 存哪些pixels是屬於該cluster的
        self.pixels = []

    def update(self, l, a, b, i, j):
        self.l, self.a, self.b, self.i, self.j = l, a, b, i, j

class SLIC(object):
    def __init__(self, img, k, m, max_iter=10):
        # 0 ≤ l ≤ 100 , −127 ≤ a ≤ 127, −127 ≤ b ≤ 127 
        self.data = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        self.k = k
        self.m = m
        self.max_iter = max_iter
        self.N = self.data.shape[0] * self.data.shape[1]
        self.S = int((self.N/self.k)**0.5)

        self.clusters = []
        self.label = {} # dictionary

        # Set distance d(i) = inf for each pixel i.
        self.dis = np.full((self.data.shape[0], self.data.shape[1]), np.inf)
    
    def make_cluster(self, i, j):
        i, j = int(i), int(j)
        return Cluster(self.data[i,j][0],
                       self.data[i,j][1],
                       self.data[i,j][2], i, j)

    # Initialize cluster centers Ck = [l_k,a_k,b_k,x_k,y_k]^T by sampling pixels at regular grid steps S.
    # 每隔 S pixel 設一個 cluster
    def init_clusters(self):
        for i in range(self.S//2, self.data.shape[0], self.S):
            for j in range(self.S//2, self.data.shape[1], self.S):
                self.clusters.append(self.make_cluster(i, j))

    # why?
    def get_gradient(self, i, j):
        if i+1 >= self.data.shape[0]:
            i = self.data.shape[0]-2
        if j+1 >= self.data.shape[1]:
            j = self.data.shape[1]-2

        gradient = self.data[i+1,j+1][0]-self.data[i,j][0] + \
                   self.data[i+1,j+1][1]-self.data[i,j][1] + \
                   self.data[i+1,j+1][2]-self.data[i,j][2]
        return gradient

    # Move cluster centers to lowest gradient pos in a 3x3 neighborhood.
    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.i, cluster.j)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    _i = cluster.i + di
                    _j = cluster.j + dj
                    new_gradient = self.get_gradient(_i, _j)
                    if new_gradient < cluster_gradient:
                        cluster.update(self.data[_i][_j][0], self.data[_i][_j][1], self.data[_i][_j][2], _i, _j)
                        cluster_gradient = new_gradient

    # 最花時間
    def assignment(self):        
        for cluster in self.clusters:
            for i in range(cluster.i - 2*self.S, cluster.i + 2*self.S):
                if i < 0 or i >= self.data.shape[0]:
                    continue
                for j in range(cluster.j - 2*self.S, cluster.j + 2*self.S):
                    if j < 0 or j >= self.data.shape[1]: 
                        continue
                    l, a, b = self.data[i,j]

                    dc = ((l-cluster.l)**2 + (a-cluster.a)**2 + (b-cluster.b)**2)**0.5
                    ds = ((i-cluster.i)**2 + (j-cluster.j)**2)**0.5
                    # print(dc, ds)
                    D = ((dc/self.m)**2 + (ds/self.S)**2)**0.5
                    # print(f'D = {D}')
                    if D < self.dis[i,j]:
                        self.dis[i,j] = D

                        # set l(i) = k
                        if (i, j) in self.label:
                            self.label[(i, j)].pixels.remove((i, j))
                        self.label[(i, j)] = cluster
                        cluster.pixels.append((i, j))

    def update_cluster(self):
        for cluster in self.clusters:
            sum_l = sum_a = sum_b = sum_i = sum_j = 0
            number = len(cluster.pixels)
            # print(f'number = {number}')

            for p in cluster.pixels:
                sum_l += self.data[p[0]][p[1]][0]
                sum_a += self.data[p[0]][p[1]][1]
                sum_b += self.data[p[0]][p[1]][2]
                sum_i += p[0]
                sum_j += p[1]

            _l, _a, _b = int(sum_l/number), int(sum_a/number), int(sum_b/number)
            _i, _j = int(sum_i/number), int(sum_j/number)
            cluster.update(_l, _a, _b, _i, _j)

    def repeat(self):
        for i in trange(self.max_iter):
            self.assignment()
            print('finish assignment')
            self.update_cluster()
            # print('finish update')

        # print('output image')
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            
            # 標記 cluster
            image_arr[cluster.i][cluster.j][0] = 0
            image_arr[cluster.i][cluster.j][1] = 0
            image_arr[cluster.i][cluster.j][2] = 0    

        bgr_arr = cv2.cvtColor(image_arr, cv2.COLOR_LAB2BGR)
        name = f'lenna_k{self.k}_m{self.m}.png'
        # Show(name, bgr_arr)
        cv2.imwrite(name, bgr_arr)

def SLIC(img, max_iter=10):
    #設定SLIC初始化設定
    slic = cv2.ximgproc.createSuperpixelSLIC(img) 
    slic.iterate(max_iter)
    mask_slic = slic.getLabelContourMask() #建立超像素的遮罩，mask_slic數值為1
    # label_slic = slic.getLabels()        #獲得超像素的標籤
    # number_slic = slic.getNumberOfSuperpixels()  #獲得超項素的數量
    mask_inv_slic = cv2.bitwise_not(mask_slic)  
    img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic) #在原圖中繪製超像素邊界
    cv2.imwrite('./SLIC.png', img_slic)    #將繪製邊界的圖片儲存

if __name__ == '__main__':
    cpu_count = mp.cpu_count()
    print(f'cpu_count = {cpu_count}')

    bgr = cv2.imread('./Lenna.png')

    print(f'N = {bgr.shape[0]} * {bgr.shape[1]} = {bgr.shape[0]*bgr.shape[1]}')

    # 10 iterations suffices for most images
    # When using the CIELAB color space, m can be in the range [1, 40].
    # p = SLIC(img=bgr, k=64, m=40, max_iter=10)
    # p.init_clusters()
    # p.move_clusters()
    # print(f'finish move_clusters')
    # p.repeat()

    # opencv 的 SLIC
    SLIC(bgr, max_iter=10)