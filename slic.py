import numpy as np
from tqdm import trange
import cv2
# import multiprocessing as mp
# import torch

# device = torch.device('mps')
# print(device)

def Show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyWindow(name)

class Cluster(object):
    cluster_index = 0
    def __init__(self, l, a, b, i, j):
        self.update(l, a, b, i, j)
        self.pixels = [] # 存哪些pixels是屬於該cluster的
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, l, a, b, i, j):
        self.l, self.a, self.b, self.i, self.j = l, a, b, i, j

class SLIC(object):
    def __init__(self, path, k, m, max_iter=10):
        # 0 ≤ l ≤ 100 , −127 ≤ a ≤ 127, −127 ≤ b ≤ 127 
        self.img = cv2.imread(path)
        self.lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        # self.lab = torch.from_numpy(cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB))
        # print(self.lab)
        self.k = k
        self.m = m
        self.max_iter = max_iter
        self.h = self.lab.shape[0]
        self.w = self.lab.shape[1]
        self.N = self.h * self.w
        self.S = int((self.N/self.k)**0.5)
        # print(f'S = {self.S}')

        self.clusters = []

        # Set label l(i) = -1 for each pixel i.
        self.label = np.full((self.h, self.w), -1)
        # Set distance d(i) = inf for each pixel i.
        self.dis = np.full((self.h, self.w), np.inf)

    # Initialize cluster centers C_k = [l_k,a_k,b_k,x_k,y_k]
    # by sampling pixels at regular grid steps S.
    def init_clusters(self):
        for i in range(self.S//2, self.h, self.S):
            for j in range(self.S//2, self.w, self.S):
                self.clusters.append(Cluster(self.lab[i,j][0],
                                             self.lab[i,j][1],
                                             self.lab[i,j][2], i, j))
    # why?
    def get_gradient(self, i, j):
        if i+1 >= self.h:
            i = self.h-2
        if j+1 >= self.w:
            j = self.w

        gradient = self.lab[i+1,j+1][0]-self.lab[i,j][0] + \
                   self.lab[i+1,j+1][1]-self.lab[i,j][1] + \
                   self.lab[i+1,j+1][2]-self.lab[i,j][2]
        # print(gradient)
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
                        cluster.update(self.lab[_i][_j][0], self.lab[_i][_j][1], self.lab[_i][_j][2], _i, _j)
                        cluster_gradient = new_gradient

    def assignment(self): # 最花時間
        for cluster in self.clusters:
            for i in range(cluster.i-self.S, cluster.i+self.S):
                if i < 0 or i >= self.h:
                    continue
                for j in range(cluster.j-self.S, cluster.j+self.S):
                    if j < 0 or j >= self.w: 
                        continue
                    l, a, b = self.lab[i,j]

                    dc = ((l-cluster.l)**2 + (a-cluster.a)**2 + (b-cluster.b)**2)**0.5
                    ds = ((i-cluster.i)**2 + (j-cluster.j)**2)**0.5
                    D = ((dc/self.m)**2 + (ds/self.S)**2)**0.5
                    if D < self.dis[i,j]:
                        self.dis[i,j] = D

                        # set l(i) = k
                        # 如果這個label已經存在其他cluster裡面了，就要把舊的刪掉
                        if self.label[i,j] != -1:
                            self.clusters[self.label[i,j]].pixels.remove((i,j))
                        self.label[i,j] = cluster.no
                        cluster.pixels.append((i, j))

    def update_cluster(self):
        for cluster in self.clusters:
            sum_l = sum_a = sum_b = sum_i = sum_j = 0
            number = len(cluster.pixels)
            # print(f'number = {number}')
            for p in cluster.pixels:
                sum_l += self.lab[p[0]][p[1]][0]
                sum_a += self.lab[p[0]][p[1]][1]
                sum_b += self.lab[p[0]][p[1]][2]
                sum_i += p[0]
                sum_j += p[1]
            _l, _a, _b = int(sum_l/number), int(sum_a/number), int(sum_b/number)
            _i, _j = int(sum_i/number), int(sum_j/number)
            cluster.update(_l, _a, _b, _i, _j)

    def repeat(self):
        for i in trange(self.max_iter):
            self.assignment()
            self.update_cluster()

        image_arr = np.copy(self.lab)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            
            # 標記 cluster
            image_arr[cluster.i][cluster.j] = [0,0,0]

        bgr_arr = cv2.cvtColor(image_arr, cv2.COLOR_LAB2BGR)
        name = f'lenna_k{self.k}_m{self.m}_loop{self.max_iter}.png'
        Show(name, bgr_arr)
        cv2.imwrite(name, bgr_arr)
    
    def draw_contour(self):
        contour = np.copy(self.img)
        di = [0, 1]
        dj = [1, 0]
        for i in range(self.h):
            for j in range(self.w):
                for k in range(len(di)):
                    if not (0 <= i+di[k] < self.h and 0 <= j+dj[k] < self.w):
                        continue
                    if self.label[i,j] != self.label[i+di[k], j+dj[k]]:
                        contour[i,j] = [0,0,0]
        name = f'lenna_k{self.k}_m{self.m}_loop{self.max_iter}_contour.png'
        Show(name, contour)
        cv2.imwrite(name, contour)
        return contour

def cv_SLIC(img, max_iter=10):
    slic = cv2.ximgproc.createSuperpixelSLIC(img) #設定SLIC初始化設定
    slic.iterate(max_iter)
    mask_slic = slic.getLabelContourMask() #建立超像素的遮罩，mask_slic數值為1
    # label_slic = slic.getLabels()        #獲得超像素的標籤
    # number_slic = slic.getNumberOfSuperpixels()  #獲得超項素的數量
    mask_inv_slic = cv2.bitwise_not(mask_slic)  
    img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic) #在原圖中繪製超像素邊界
    cv2.imwrite('./SLIC.png', img_slic)    #將繪製邊界的圖片儲存

if __name__ == '__main__':
    # 10 iterations suffices for most images
    # When using the CIELAB color space, m can be [1, 40].
    # k=4096, 512, 
    p = SLIC(path='./Lenna.png', k=1024, m=40, max_iter=5)
    p.init_clusters()
    p.move_clusters()
    # print(f'finish move_clusters')
    p.repeat()
    label = p.label
    # print(label)
    contour = p.draw_contour()

    # opencv 的 SLIC
    # cv_SLIC(bgr, max_iter=10)