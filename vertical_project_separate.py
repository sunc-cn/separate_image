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
    # enhance image's black pixel
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
    im_pixel_dict = {}
    for x in range(len(labels)):
        for y in range(len(labels[x])):
            if labels[x][y] != 0:
                k = labels[x][y]
                im_dict[k][x][y][0] = raw_im_processed_array[x][y][0]
                im_dict[k][x][y][1] = raw_im_processed_array[x][y][1]
                im_dict[k][x][y][2] = raw_im_processed_array[x][y][2]
                if k not in im_pixel_dict.keys():
                    im_pixel_dict[k] = 1
                else:
                    im_pixel_dict[k] += 1
                pass
    im_lists = []
    minimum_pixels = 15
    for (k,v) in im_dict.items():
        if k != 0:
            if im_pixel_dict[k] < minimum_pixels:
                # filter overlow pixels
                continue
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
        #im = im.filter(ImageFilter.MaxFilter(3))
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
    #check_im_lists(im_lists)
    double_list = find_double_with_img(im_lists)
    if len(double_list) != 0:
        index_dict = {}
        for index in double_list:
            im = im_lists[index]
            (wb,we) = get_im_width_ends(im)
            start_point = (0,int((we-wb)/2 + wb))
            im_grey = convert_2_gray(np.array(im))
            drop_path_points = drop_fall_to_down(start_point,im_grey)
            ims = apply_split_path(im,drop_path_points)
            index_dict[index] = ims
        new_im_lists = []
        for index in range(len(im_lists)):
            if index in double_list:
                ims = index_dict[index]
                new_im_lists.append(ims[0])
                new_im_lists.append(ims[1])
                pass
            else:
                new_im_lists.append(im_lists[index])
        im_lists = new_im_lists
        pass
    #for index in range(len(im_lists)):
    #    im = im_lists[index]
    #    im.save(str(index)+".bmp")
    return im_lists
    pass

def apply_split_path(im,split_path_points):
    raw_array = np.array(im)
    im_array1 = np.zeros(shape=raw_array.shape,dtype="uint8")
    im_array1.fill(255)
    im_array2 = np.zeros(shape=raw_array.shape,dtype="uint8")
    im_array2.fill(255)
    (h,w,_)  = raw_array.shape
    for p in split_path_points:
        (px,py) = p
        for y in range(py):
            im_array1[px][y] = raw_array[px][y]
        for y in range(py+1,w):
            im_array2[px][y] = raw_array[px][y]
    im1 = matrix_2_image(im_array1)
    im2 = matrix_2_image(im_array2)
    #im1.save("im1.bmp")
    #im2.save("im2.bmp")
    return (im1,im2)
    pass

def get_im_height_width(im):
    im_array = np.array(im)
    (height,width,_) = im_array.shape
    white = np.array([255,255,255])
    height_be = 0
    height_ee = height - 1
    for x in range(height):
        is_found = False
        for y in range(width):
            if not (im_array[x][y] ==  white).all():
                height_be = x
                is_found = True
                break
        if is_found:
            break
    for x in range(height-1,-1,-1):
        is_found = False
        for y in range(width):
            if not (im_array[x][y] ==  white).all():
                height_ee = x
                is_found = True
                break
        if is_found:
            break
    
    width_be = 0
    width_ee = width - 1
    for y in range(width):
        is_found = False
        for x in range(height):
            if not (im_array[x][y] ==  white).all():
                width_be = y
                is_found = True
                break
        if is_found:
            break
    for y in range(width-1,-1,-1):
        is_found = False
        for x in range(height):
            if not (im_array[x][y] ==  white).all():
                width_ee = y
                is_found = True
                break
        if is_found:
            break
    #print(height_be,height_ee,width_be,width_ee)  
    h = height_ee - height_be + 1
    w = width_ee - width_be + 1
    #print(h,w)
    return (h,w) 
    pass

def check_im_lists(im_lists):
    hw_list = []
    for item in im_lists:
        hw = get_im_height_width(item)
        hw_list.append(hw)
    if len(hw_list) == 0:
        return True
    height_list = []
    width_list = []
    for item in hw_list:
        height_list.append(item[0])
        width_list.append(item[1])
    height_list = sorted(height_list)
    width_list = sorted(width_list)
    #print(height_list)
    #print(width_list)
    h_min = height_list[0]
    h_max = height_list[len(height_list)-1]
    w_min = width_list[0]
    w_max = width_list[len(width_list)-1]
    h_mid_index = int(len(height_list)/2)
    w_mid_index = int(len(width_list)/2)
    h_mid = height_list[h_mid_index]
    w_mid = width_list[w_mid_index]
    w_limit = 4
    h_limit = 4
    if w_mid - w_min > w_limit or w_max - w_mid > w_limit:
        print("width analyze",w_min,w_mid,w_max)
        return False
    if h_mid - h_min > h_limit or h_max-h_mid > h_limit:
        print("height analyze",h_min,h_mid,h_max)
        return False
    return True
    pass

def drop_fall_to_down(start_point,im_grey):
    (height,width) = im_grey.shape
    (cx,cy) = start_point
    curr = start_point
    points = []
    while cx < height-2:
        left1 = (cx,cy-1)
        right3 = (cx,cy+1)
        r_left1 = (cx+1,cy-1)
        r_mid2= (cx+1,cy)
        r_right3 = (cx+1,cy+1)
        r1_left1 = (cx+2,cy-1)
        r1_mid2= (cx+2,cy)
        r1_right3 = (cx+2,cy+1)
        if im_grey[r_mid2[0]][r_mid2[1]] == 255:
            curr = r_mid2
        elif im_grey[r_left1[0]][r_left1[1]] == 255:
            curr = r_left1
        elif im_grey[r_right3[0]][r_right3[1]] == 255:
            curr = r_right3
        elif im_grey[r1_right3[0]][r1_right3[1]] == 255:
            curr = r_mid2
        elif im_grey[r1_left1[0]][r1_left1[1]] == 255:
            curr = r_mid2
        elif im_grey[r1_left1[0]][r1_left1[1]] == 255:
            curr = r_mid2
        elif im_grey[left1[0]][left1[1]] == 255:
            curr = left1 
        elif im_grey[right3[0]][right3[1]] == 255:
            curr = right3
        else:
            curr = r_mid2
        if curr in points:
            curr = r_mid2
        points.append(curr)
        cx = curr[0]
        cy = curr[1]
        #print("down,cx",cx,curr,cy)
    return points

def get_im_width_ends(im):
    im_array = np.array(im)
    (height,width,_) = im_array.shape
    white = np.array([255,255,255])
    width_be = 0
    width_ee = width - 1
    for y in range(width):
        is_found = False
        for x in range(height):
            if not (im_array[x][y] ==  white).all():
                width_be = y
                is_found = True
                break
        if is_found:
            break
    for y in range(width-1,-1,-1):
        is_found = False
        for x in range(height):
            if not (im_array[x][y] ==  white).all():
                width_ee = y
                is_found = True
                break
        if is_found:
            break
    return (width_be,width_ee)

def find_double_with_img(im_lists):
    hw_list = []
    for item in im_lists:
        hw = get_im_height_width(item)
        hw_list.append(hw)
    if len(hw_list) == 0:
        return True
    width_list = []
    for item in hw_list:
        width_list.append(item[1])
    width_list_sorted = sorted(width_list)
    w_mid_index = int(len(width_list)/2)
    w_mid = width_list_sorted[w_mid_index]
    width_prop = 1.6
    double_width_list = []
    for index in range(len(width_list)):
        w = width_list[index]
        if (float(w)/float(w_mid)) > width_prop:
            double_width_list.append(index)
    print(double_width_list)
    return double_width_list

if __name__ == "__main__":
    #hybird_separate("./khz.jpg")
    #hybird_separate_ex("./khz.jpg")
    #im_file = "./100458.jpg"
    #im_file = "./201032792552.jpg"
    im_file = "./dst2.png"
    #hybird_separate(im_file)
    hybird_separate_ex(im_file)

