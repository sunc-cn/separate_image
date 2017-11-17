import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

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
##############################################################################
# Binarize image data

im = np.array(Image.open('./1RIA.jpg'))
im = convert2gray(im)
h, w = im.shape
X = [(h - x, y) for x in range(h) for y in range(w) if im[x][y]]
X = np.array(X)
n_clusters = 4

##############################################################################
# Compute clustering with KMeans

k_means = KMeans(init='k-means++', n_clusters=n_clusters)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

##############################################################################
# Plot result

colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF3300']
#plt.figure()
#plt.hold(True)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 1], X[my_members, 0], 'w',
            markerfacecolor=col, marker='.')
    plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('KMeans')    
plt.grid(True)
plt.savefig("./xx.jpg")
#plt.show()
