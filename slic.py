import numpy as np
from tqdm import trange
import cv2
import sys
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
        self.clustter_amount = -1

        # Set label l(i) = -1 for each pixel i.
        self.label = np.full((self.h, self.w), -1)
        # Set distance d(i) = inf for each pixel i.
        self.dis = np.full((self.h, self.w), np.inf)

    # Initialize cluster centers C_k = [l_k,a_k,b_k,x_k,y_k]
    # by sampling pixels at regular grid steps S.
    
    def init_clusters(self):
        for i in range(self.S//2, self.h, self.S):
            for j in range(self.S//2, self.w, self.S):
                if j == self.w:
                    print("error")
                self.clusters.append(Cluster(self.lab[i,j][0],
                                             self.lab[i,j][1],
                                             self.lab[i,j][2], i, j))
        self.clustter_amount=len(self.clusters)
    # why?
    def get_gradient(self, i, j, pad_img, sobel_mask_Gx, sobel_mask_Gy):
        
        region = pad_img[i:i+3, j:j+3]
        
        intensity = np.sqrt(region[:,:,0]**2+region[:,:,1]**2+region[:,:,2]**2)
        #print(intensity.shape)
        gradient_Gx = (intensity*sobel_mask_Gx).sum()
        gradient_Gy = (intensity*sobel_mask_Gy).sum()
        gradient = np.sqrt(gradient_Gx**2 + gradient_Gy**2)
        return gradient

    # Move cluster centers to lowest gradient pos in a 3x3 neighborhood.
    def move_clusters(self):
        pad_img = np.pad(self.lab, 1, 'edge')
        sobel_mask_Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/4
        sobel_mask_Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/4
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.i, cluster.j, pad_img, sobel_mask_Gx,sobel_mask_Gy)
            upmost = max(cluster.i-1, 0)
            downmost = min(cluster.i+2, self.h)
            leftmost = max(cluster.j-1, 0)
            rightmost = min(cluster.j+2, self.w)
            
            for _i in range(upmost, downmost):
                for _j in range(leftmost, rightmost):
                    new_gradient = self.get_gradient(_i, _j, pad_img, sobel_mask_Gx,sobel_mask_Gy)
                    if new_gradient < cluster_gradient:
                        cluster.update(self.lab[_i][_j][0], self.lab[_i][_j][1], self.lab[_i][_j][2], _i, _j)
                        cluster_gradient = new_gradient

    def assignment(self): # 最花時間
        for cluster in self.clusters:
            #define avalible region
            upmost = max(cluster.i-self.S, 0)
            downmost = min(cluster.i+self.S, self.h)
            leftmost = max(cluster.j-self.S, 0)
            rightmost = min(cluster.j+self.S, self.w)
            #get region and caculate dc, ds and D
            region = self.lab[upmost:downmost, leftmost:rightmost].astype('int32')
            dc=(region-np.array([cluster.l, cluster.a, cluster.b], dtype='int32'))**2
            dc = (dc[:,:,0]+dc[:,:,1]+dc[:,:,2])
            yy, xx = np.mgrid[upmost:downmost, leftmost:rightmost]
            ds = (yy-cluster.i)**2+(xx-cluster.j)**2
            D = (dc+ds*(self.m/self.S)**2)**0.5
            #update distance
            dis_region = self.dis[upmost:downmost, leftmost:rightmost]
            self.dis[upmost:downmost, leftmost:rightmost] = np.minimum(dis_region, D)
            #update label
            label = self.label[upmost:downmost, leftmost:rightmost]
            label[D<=dis_region] = cluster.no
            self.label[upmost:downmost, leftmost:rightmost] = label
        
        
        
        pixels = []
        
        for i in range(self.clustter_amount):
            pixels.append([])
            
        
        for i in range(self.h):
            for j in range(self.w):
                #print(i, j, self.label.shape, self.label[i,j], len(pixels))
                pixels[self.label[i,j]].append((i,j))
    
        for i in range(self.clustter_amount):
            if len(pixels[i])==0:
                print('error', i)
            self.clusters[i].pixels = pixels[i]
                
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
        #check for pixel that is surround by other label
        pad_label = np.pad(self.label, 1, 'constant', constant_values=-1)
        for i in range(self.h):
            for j in range(self.w):
                if pad_label[i+1,j+1]!=pad_label[i, j+1] and pad_label[i+1,j+1]!=pad_label[i+1, j] and pad_label[i+1,j+1]!=pad_label[i+2, j+1] and pad_label[i+1,j+1]!=pad_label[i+1, j+2]:
                    dis = []
                    for k in [pad_label[i, j+1], pad_label[i+1, j], pad_label[i+2, j+1], pad_label[i+1, j+2]]:
                        if k == -1:
                            continue
                        else:
                            dis.append((i-self.clusters[k].i)**2+(j-self.clusters[k].j)**2)
                    self.label[i,j] = dis.index(min(dis))
                            
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
    
    def draw_contour(self, name):
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
        #name = f'lenna_k{self.k}_m{self.m}_loop{self.max_iter}_contour.png'
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
    path=sys.argv[1]
    out = sys.argv[2]
    k = int(sys.argv[3])
    m = int(sys.argv[4])
    
    
    p = SLIC(path=path, k=k, m=m, max_iter=10)
    p.init_clusters()
    p.move_clusters()
    print(f'finish move_clusters')
    #p.assignment()
    p.repeat()
    label = p.label
    #print(label)
    contour = p.draw_contour(out)

    # opencv 的 SLIC
    # cv_SLIC(bgr, max_iter=10)
