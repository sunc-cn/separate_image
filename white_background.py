#encoding=utf-8

from PIL import Image  
from PIL import ImageFilter  
import numpy as np
from sklearn.cluster import KMeans
import os

def matrix_2_image(data):
    data = data
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

# the coordinate is (height,width)
def get_split_pixel(im,n_clusters):
    im_array = np.array(im)
    X = []
    (h,w,_) = im_array.shape
    for x in range(h):
        for y in range(w):
            rgb = im_array[x][y]
            X.append((rgb[0],rgb[1],rgb[2]))
    X = np.array(X)
    # Compute clustering with KMeans
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    center_list = []
    for k in range(n_clusters):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        center_list.append(cluster_center)
    sorted_list = sorted(center_list,key=lambda x:x[0]+x[1]+x[2])
    print(sorted_list)
    if len(sorted_list) == n_clusters and n_clusters > 2:
        pixel1 = sorted_list[0]
        pixel2 = sorted_list[1]
        #split_pixel = ([(pixel1[0]+pixel2[0])/2,(pixel1[1]+pixel2[1])/2,(pixel1[1]+pixel2[1])/2])
        split_pixel = pixel2
        return split_pixel
    else:
        raise NameError("sorted_list size is not equal with n_clusters")
def apply_split_pixel(im,split_pixel):
    im_array = np.array(im)
    X = []
    (h,w,_) = im_array.shape
    new_item = np.zeros(im_array.shape,dtype="uint8")
    new_item.fill(255)
    for x in range(h):
        for y in range(w):
            rgb = im_array[x][y]
            if rgb[0] <= split_pixel[0] and rgb[1] <= split_pixel[1] and rgb[2] <= split_pixel[2]:
                new_item[x][y] = rgb
    new_im = matrix_2_image(new_item)
    return new_im
def white_background(im_path):
    im = Image.open(im_path)
    pixel = get_split_pixel(im,4)
    new_im = apply_split_pixel(im,pixel)
    new_im = new_im.filter(ImageFilter.SMOOTH_MORE)
    return new_im

if __name__ == "__main__":
    file_name = "./id_number.png"
    #file_name = "./dst2.png"
    new_im = white_background(file_name)
    new_im.save("./new_id_number.bmp")


    
