import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        #r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def separate_image(raw_file_name):
    file_name = os.path.basename(raw_file_name)
    raw_name,ext_name = os.path.splitext(file_name)
    if len(raw_name ) != 4:
        print('wrong raw_name:',raw_name)
        exit(-1) 
    ##############################################################################
    # Binarize image data
    raw_image = np.array(Image.open(raw_file_name))
    #print("raw_image:",raw_image.shape,raw_image.dtype)
    gray_image = convert2gray(raw_image)
    h, w = gray_image.shape
    im = gray_image
    #print(h,w)
    X = [(h - x, y) for x in range(h) for y in range(w) if im[x][y] and x > 5 and y > 3]
    X = np.array(X)
    n_clusters = 4
    #print("x",type(X),X.shape,X.ndim)
    
    ##############################################################################
    # Compute clustering with KMeans
    
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    
    ##############################################################################
    # Plot result
    #colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF3300']
    colors = ['#000000', '#000000', '#000000', '#000000']
    center_tuple_list = []
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        #print("center",type(cluster_center),cluster_center)
        center_tuple_list.append((k,cluster_center[1]))
    #print(center_tuple_list)
    sorted_list = sorted(center_tuple_list,key=lambda x:x[1])
    #print(sorted_list)
    map_dict = {}
    index = 0
    map_index_dict = {}
    for item in sorted_list:
        (k,_) = item
        map_dict[k] = raw_name[index]
        map_index_dict[k] = index
        index += 1
    
    for k, col in zip(range(n_clusters), colors):
        #print("k,labels",k,k_means_labels)
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        #print(type(X),X.shape,X.ndim)
        plt.plot(X[my_members, 1], X[my_members, 0], 'w',
                                markerfacecolor=col, marker='.')
        blank_item = np.zeros(shape=(32,90,3),dtype="uint8")
        #print(my_members) 
        for item in X[my_members]:
            blank_item[item[0],item[1],0] = raw_image[item[0],item[1],0]
            blank_item[item[0],item[1],1] = raw_image[item[0],item[1],1]
            blank_item[item[0],item[1],2] = raw_image[item[0],item[1],2]
        toi = MatrixToImage(blank_item)
        toi.save(raw_name +"_"+ str(map_index_dict[k]) +"_" + map_dict[k] + ".jpg")



if __name__ == "__main__":
    file_name = "./lqhr.jpg"
    separate_image(file_name)
    pass
