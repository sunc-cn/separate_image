#encoding=utf-8

from PIL import Image  
from PIL import ImageFilter  
import numpy as np
import os
import skimage
import skimage.measure
import skimage.morphology

K_Max_Height = 32
K_Dst_Height = 32
K_Dst_Width = 32

def convert_2_gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        #r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def matrix_2_image(data):
    data = data
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def preprocess_im(im):
    #im = im.filter(ImageFilter.MinFilter(1))  
    #im = im.filter(ImageFilter.CONTOUR)
    #im = im.filter(ImageFilter.SHARPEN)  
    #im = im.filter(ImageFilter.EDGE_ENHANCE)
    #im = im.filter(ImageFilter.SMOOTH)
    #im = im.filter(ImageFilter.BLUR)
    #im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #im = im.filter(ImageFilter.MinFilter(3))  

    #im.save("new.png")
    raw_array = np.array(im)
    new_item = np.zeros(shape=raw_array.shape,dtype="uint8")
    new_item.fill(255)
    (h,w,c) = raw_array.shape
    for x in range(h):
        for y in range(w):
            if raw_array[x][y][0] < 170 and raw_array[x][y][1] < 170 and raw_array[x][y][2] < 170:
                new_item[x][y][0] = 0
                new_item[x][y][1] = 0
                new_item[x][y][2] = 0
    new_im = matrix_2_image(new_item)
    #new_im = im.filter(ImageFilter.SMOOTH_MORE)
    #new_im.save("new2.png")
    return new_im
    pass

def connection_separate(im_path):
    # TODO apply some image filter
    raw_im = Image.open(im_path)
    raw_im_processed = preprocess_im(raw_im)
    raw_im_processed_array = np.array(raw_im_processed)
    raw_bin = raw_im_processed.convert("1")
    raw_bin_array = np.array(raw_bin)
    #raw_bin_array = skimage.morphology.remove_small_objects(np.array(raw_bin),min_size=1111,connectivity=1,in_place=False)
    #labels,num = skimage.measure.label(np.array(raw_bin),background=1,connectivity=2,neighbors=4,return_num=True)
    labels,num = skimage.measure.label(raw_bin_array,background=1,connectivity=2,neighbors=8,return_num=True)
    #print(labels,labels.shape,num,labels.max())
    im_dict = {}
    raw_array = np.array(raw_im)
    for i in range(num+1):
        new_item = np.zeros(shape=raw_array.shape,dtype="uint8")
        new_item.fill(255)
        im_dict[i] = new_item
    for x in range(len(labels)):
        for y in range(len(labels[x])):
            if labels[x][y] != 0:
                k = labels[x][y]
                im_dict[k][x][y][0] = raw_im_processed_array[x][y][0]
                im_dict[k][x][y][1] = raw_im_processed_array[x][y][1]
                im_dict[k][x][y][2] = raw_im_processed_array[x][y][2]
                pass
    im_lists = []
    for (k,v) in im_dict.items():
        if k != 0:
            im = matrix_2_image(v)
            #im.save(str(k)+".png")
            im_lists.append(im)
        pass
    return im_lists
        
def vertical_separate(im_lists):
    split_im_arrays = []
    for im_item in im_lists:
        raw_bin = im_item.convert("1")
        raw_image = np.array(raw_bin)
        gray_image = raw_image
        h, w = gray_image.shape
        if h > K_Max_Height:
            print("incorrect height:%d"%(h))
            return split_im_arrays
        im =  gray_image

        last_blank = True
        row_index_list = []
        for y in range(w):
            is_blank = True
            for x in range(h):
                if im[x][y] == False:
                    is_blank = False
            if last_blank != is_blank:
                row_index_list.append(y)
            last_blank = is_blank
        #print(row_index_list,len(row_index_list))
        if len(row_index_list) <= 1:
            return split_im_arrays
        if len(row_index_list)%2 != 0:
            row_index_list.append(w-1)
        for i in range(1,len(row_index_list),2):
            left_w = row_index_list[i-1]
            right_w = row_index_list[i]
            if right_w-left_w+1 > K_Dst_Width:
                print(right_w-left_w+1," is bigger than ",K_Dst_Width)
                return
            sub_im = np.zeros(shape=(K_Dst_Height,K_Dst_Width,3),dtype="uint8")
            sub_im.fill(255)
            x_delta = K_Dst_Height - 1
            for x in range(h-1,-1,-1):
                x_delta -= 1
                for y in range(left_w,right_w +1):
                    #print(x,y,im[x][y])
                    if im[x][y] == False:
                        sub_im[x_delta][y-left_w+5][0] = 0
                        sub_im[x_delta][y-left_w+5][1] = 0
                        sub_im[x_delta][y-left_w+5][2] = 0
            #sub_im_x = matrix_2_image(sub_im)
            #sub_im_x.save(str(left_w)+".jpg")
            split_im_arrays.append((left_w,sub_im))
    return split_im_arrays
    pass

def save_separated_ims(split_im_arrays):
    if len(split_im_arrays) == 0:
        print("save_separated_ims length ",len(split_im_arrays))
        return
    sorted_list = sorted(split_im_arrays,key=lambda x:x[0])
    index = 0
    for item in sorted_list:
        (_,im_array) = item
        im = matrix_2_image(im_array) 
        im = im.filter(ImageFilter.SMOOTH_MORE)
        #im.save(str(index)+".jpg","jpeg",quality=100)
        im.save(str(index)+".bmp")
        index += 1
    pass 

def fetch_separated_ims(split_im_arrays):
    sorted_list = sorted(split_im_arrays,key=lambda x:x[0])
    im_lists = []
    for item in sorted_list:
        (_,im_array) = item
        im = matrix_2_image(im_array) 
        im = im.filter(ImageFilter.SMOOTH_MORE)
        im_lists.append(im)
    return im_lists
    pass 
            
def hybird_separate(im_path):
    im_lists = connection_separate(im_path)
    split_im_arrays = vertical_separate(im_lists)
    save_separated_ims(split_im_arrays)
    pass

def hybird_separate_ex(im_path):
    im_lists = connection_separate(im_path)
    split_im_arrays = vertical_separate(im_lists)
    im_lists = fetch_separated_ims(split_im_arrays)
    return im_lists
    pass

if __name__ == "__main__":
    hybird_separate("./khz.jpg")
