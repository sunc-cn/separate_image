#encoding=utf-8

from PIL import Image  
from PIL import ImageFilter  
import numpy as np
from sklearn.cluster import KMeans
import os

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

def remove_invalid_pixel(im_path):
    raw_image = np.array(Image.open(im_path))
    gray_image = convert_2_gray(raw_image)
    h, w = gray_image.shape
    im = gray_image
    new_item = np.zeros(shape=raw_image.shape,dtype="uint8")
    new_item.fill(255)
    for x in range(32):
        for y in range(90):
            if x >3 and y >5:
                c1 = 255
                c2 = 255
                c3 = 255
                if gray_image[x,y] < 128:
                    c1 = 0
                if gray_image[x,y] < 128:
                    c2 = 0
                if gray_image[x,y] < 128:
                    c3 = 0
                new_item[x,y,0] = c1 
                new_item[x,y,1] = c2 
                new_item[x,y,2] = c3
    im = matrix_2_image(new_item) 
    return im
    pass

# the coordinate is (height,width)
def get_kmean_center(im_grey,n_clusters):
    gray_image = im_grey
    h, w = gray_image.shape
    im = gray_image
    X = [(h - x, y) for x in range(h) for y in range(w) if not im[x][y]]
    X = np.array(X)
    n_clusters = 4
    # Compute clustering with KMeans
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    center_tuple_list = []
    for k in range(n_clusters):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        #print("center",type(cluster_center),cluster_center)
        center_tuple_list.append((cluster_center[0],cluster_center[1]))
    #print(center_tuple_list)
    sorted_list = sorted(center_tuple_list,key=lambda x:x[1])
    #print(sorted_list)
    return sorted_list

def get_drop_fall_init_point(left_coor,right_coor,im_grey):
    #print(im_grey.shape)
    (height,width) = im_grey.shape
    (lx,ly) = left_coor
    (rx,ry) = right_coor
    w_start = int(ly) - 1
    w_end = int(ry) + 1
    if w_end > width -1 :
        w_end -= 1
    if w_start < 0:
        w_start = 0
    white_points = []
    #print(w_start,w_end)
    for h in range(height):
        for w in range(w_start,w_end):
            if w-1 >= 0 and w+1 < width:
                left = im_grey[h][w-1]
                curr = im_grey[h][w]
                right = im_grey[h][w+1]
                if left == 0 and right == 0 and curr == 255:
                    white_points.append((h,w))
    #print(white_points)
    if len(white_points) == 0:
        for h in range(height):
            for w in range(w_start,w_end):
                if w-1 >= 0 and w+1 < width and w+2 < width:
                    left = im_grey[h][w-1]
                    curr = im_grey[h][w]
                    right1 = im_grey[h][w+1]
                    right2 = im_grey[h][w+2]
                    if left == 0 and right2 == 0 and curr == 255 and right1 == 255:
                        white_points.append((h,w))
    if len(white_points) == 0:
        for h in range(height):
            for w in range(w_start,w_end):
                if w-1 >= 0 and w+1 < width and w+2 < width and w+3 < width:
                    left = im_grey[h][w-1]
                    curr = im_grey[h][w]
                    right1 = im_grey[h][w+1]
                    right2 = im_grey[h][w+2]
                    right3 = im_grey[h][w+3]
                    if left == 0 and right3 == 0 and curr == 255 and right1 == 255 and right2 == 255:
                        white_points.append((h,w))
    return white_points
    pass

def calc_manhattan_distance(lhs_point,rhs_point):
    (lx,ly) = lhs_point
    (rx,ry) = rhs_point
    #dis = max(int(abs(rx-lx)),int(abs(ry-ly)))
    dis = int(abs(ry-ly))
    return dis

def pick_priority_start_point(left_coor,right_coor,init_points):
    if len(init_points) == 0:
        return (0,0)
    if len(init_points) == 1:
        return init_points[0]
    dis_list = []
    index = 0
    for item in init_points:
        (lx,ly) = left_coor
        (rx,ry) = right_coor
        (avg_x,avg_y) = (int((lx+rx)/2),int((ly+ry)/2))
        mid_point = (avg_x,avg_y)
        l = calc_manhattan_distance(item,mid_point)
        dis_list.append((index,l))
        index += 1
    #print(dis_list)
    sorted_list = sorted(dis_list,key=lambda x:x[1])
    first_item = sorted_list[0]
    (item_index,_) = first_item
    point = init_points[item_index]
    return point

def detect_down_direction(left_coor,right_coor,start_point):
    (lx,ly) = left_coor
    (rx,ry) = right_coor
    (cx,cy) = start_point
    max_x = max(rx,lx)
    if cx >= max_x: # magic number,mid line
        return False
    return True

def detect_blank_space(coordinate,im_grey,down_direction=True):
    (height,width) = im_grey.shape
    (h_coor,w_coor) = coordinate
    h_start = int(h_coor)
    h_end = height - 1
    h_step = 1
    if not down_direction:
        h_start -= 1
        h_end = 0
        h_step = -1
    else:
        h_start += 1
    w_start = int(w_coor) - 2
    w_end = int(w_coor) + 3
    if w_start < 0:
        w_start = 0
    if w_end > width -1 :
        w_end = width -1
    left_white = []
    right_white = []
    h_break = -1
    #print(h_start,h_end,h_step,coordinate)
    for h in range(h_start,h_end,h_step):
        for w in range(w_start,w_end):
            if im_grey[h][w] == 255:
                if w <= w_coor:
                    left_white.append((h,w))
                else:
                    right_white.append((h,w))
                if h_break == -1:
                    if h_step == 1:
                        h_break = h + 2
                    else:
                        h_break = h - 2
            if h_break != -1:
                if h == h_break:
                    break
    #print(left_white,right_white)
    reverse = False
    if not down_direction:
        reverse = True
    if len(left_white) >= len(right_white):
        sorted_list = sorted(left_white,key=lambda x:x[0],reverse=reverse)
        return sorted_list[0] 
    elif len(right_white) > len(left_white):
        sorted_list = sorted(right_white,key=lambda x:x[0],reverse=reverse)
        return sorted_list[0] 

def drop_fall_to_up(start_point,im_grey):
    (height,width) = im_grey.shape
    (cx,cy) = start_point
    curr = start_point
    points = []
    while cx > 2:
        left1 = (cx,cy-1)
        right3 = (cx,cy+1)
        r_left1 = (cx-1,cy-1)
        r_mid2= (cx-1,cy)
        r_right3 = (cx-1,cy+1)
        r1_left1 = (cx-2,cy-1)
        r1_mid2= (cx-2,cy)
        r1_right3 = (cx-2,cy+1)
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
        #print("up,cx",cx,cy)
    return points

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
    pass

def generate_split_path(start_point,end_point,down_direction=True):
    (sx,sy) = start_point
    (ex,ey) = end_point
    h_step = 1
    h_end = ex + 1
    if not down_direction:
        h_step = -1
        h_end = ex-1
    path_points = []
    for h in range(sx,h_end,h_step):
        path_points.append((h,sy))
    w_start = min(sy,ey)
    w_end = max(sy,ey)
    for w in range(w_start,w_end+1):
        path_points.append((ex,w))
    #print(path_points)
    return path_points
    pass

def apply_split_path(im,split_path_points):
    raw_array = np.array(im)
    for item in split_path_points:
        (x,y) = item
        raw_array[x,y,0] = 255
        raw_array[x,y,1] = 255
        raw_array[x,y,2] = 255
    im_new = matrix_2_image(raw_array) 
    return im_new

def smart_separate(file_name):
    im = remove_invalid_pixel(file_name)
    im_filter = im.filter(ImageFilter.MaxFilter(3))  
    im_grey = convert_2_gray(np.array(im))
    im_filter_grey = convert_2_gray(np.array(im_filter))
    center_list = get_kmean_center(im_filter_grey,4)
    #print(center_list)
    all_split_path_points = []
    for i in range(1,len(center_list)):
        left_coor = center_list[i-1]
        right_coor = center_list[i]
        white_points = get_drop_fall_init_point(left_coor,right_coor,im_grey)
        #print("left,right",left_coor,right_coor)
        #print("points candidate",white_points)
        start_point = pick_priority_start_point(left_coor,right_coor,white_points)  
        print("start_point ",start_point)
        if start_point == (0,0):
            print(center_list)
            print(white_points,left_coor,right_coor)
            print(start_point)
            return False
        direction = detect_down_direction(left_coor,right_coor,start_point)
        dst_point = detect_blank_space(start_point,im_grey,direction)
        split_points = generate_split_path(start_point,dst_point,direction)
        if direction:
            drop_points1 = drop_fall_to_down(dst_point,im_grey)
            drop_points2 = drop_fall_to_up(start_point,im_grey)
            all_split_path_points.extend(drop_points1)
            all_split_path_points.extend(drop_points2)
        else:
            drop_points1 = drop_fall_to_down(start_point,im_grey)
            drop_points2 = drop_fall_to_up(dst_point,im_grey)
            all_split_path_points.extend(drop_points1)
            all_split_path_points.extend(drop_points2)
            
        #print("split_points ",split_points)
        all_split_path_points.extend(split_points)
    im_split = apply_split_path(im,all_split_path_points)
    return im_split

def init_two_end_boundary(width,height):
    l_dict = {}
    r_dict = {}
    for i in range(height):
        l_dict[i] = 0
        r_dict[i] = width - 1
    return l_dict,r_dict

def get_two_side_boundary(split_points):
    p_dict = {}
    for item in split_points:
        (x,y) = item
        if x not in p_dict.keys():
            p_dict[x] = [y]
        else:
            temp = p_dict[x]
            temp.append(y)
            p_dict[x] = temp
    l_dict = {}
    r_dict = {}
    for (k,v) in p_dict.items():
        if len(v) == 1:
            l_dict[k] = v[0]
            r_dict[k] = v[0]
        else:
            v_sorted = sorted(v)
            l_dict[k] = v_sorted[0]
            r_dict[k] = v_sorted[len(v)-1]
    return l_dict,r_dict

def smart_separate_ex(file_name):
    im = remove_invalid_pixel(file_name)
    im_filter = im.filter(ImageFilter.MaxFilter(3))  
    im_grey = convert_2_gray(np.array(im))
    im_filter_grey = convert_2_gray(np.array(im_filter))
    center_list = get_kmean_center(im_filter_grey,4)
    #print(center_list)
    all_split_path_points = []
    for i in range(1,len(center_list)):
        left_coor = center_list[i-1]
        right_coor = center_list[i]
        white_points = get_drop_fall_init_point(left_coor,right_coor,im_grey)
        #print("left,right",left_coor,right_coor)
        #print("points candidate",white_points)
        start_point = pick_priority_start_point(left_coor,right_coor,white_points)  
        #print("start_point ",start_point)
        if start_point == (0,0):
            print(center_list)
            print(white_points,left_coor,right_coor)
            print(start_point)
            return None 
        direction = detect_down_direction(left_coor,right_coor,start_point)
        dst_point = detect_blank_space(start_point,im_grey,direction)
        split_points = generate_split_path(start_point,dst_point,direction)
        if direction:
            drop_points1 = drop_fall_to_down(dst_point,im_grey)
            drop_points2 = drop_fall_to_up(start_point,im_grey)
            split_points.extend(drop_points1)
            split_points.extend(drop_points2)
        else:
            drop_points1 = drop_fall_to_down(start_point,im_grey)
            drop_points2 = drop_fall_to_up(dst_point,im_grey)
            split_points.extend(drop_points1)
            split_points.extend(drop_points2)
            
        #print("split_points ",split_points)
        all_split_path_points.append(split_points)
    raw_name,_ = os.path.splitext(os.path.basename(file_name))
    item1_name = raw_name + "_0_" + raw_name[0] + ".jpg" 
    item2_name = raw_name + "_1_" + raw_name[1] + ".jpg"
    item3_name = raw_name + "_2_" + raw_name[2] + ".jpg"
    item4_name = raw_name + "_3_" + raw_name[3] + ".jpg"

    (b1,b8) = init_two_end_boundary(90,32)
    (b2,b3) = get_two_side_boundary(all_split_path_points[0])
    (b4,b5) = get_two_side_boundary(all_split_path_points[1])
    (b6,b7) = get_two_side_boundary(all_split_path_points[2])

    item1_img = apply_split_boundaries(im,b1,b2)
    item2_img = apply_split_boundaries(im,b3,b4)
    item3_img = apply_split_boundaries(im,b5,b6)
    item4_img = apply_split_boundaries(im,b7,b8)

    dst_list = []
    dst_list.append((item1_name,item1_img))
    dst_list.append((item2_name,item2_img))
    dst_list.append((item3_name,item3_img))
    dst_list.append((item4_name,item4_img))

    return dst_list

def apply_split_boundaries(im,left_boundary,right_boundary):
    #print(right_boundary)
    new_item = np.zeros(shape=(32,90,3),dtype="uint8")
    new_item.fill(255)
    raw_array = np.array(im)
    for (k,v) in left_boundary.items():
        if k in right_boundary.keys():
            w_left = v + 1
            w_right = right_boundary[k]
            #print(w_left,w_right)
            for w in range(w_left,w_right):
                new_item[k,w,0] = raw_array[k,w,0]
                new_item[k,w,1] = raw_array[k,w,1]
                new_item[k,w,2] = raw_array[k,w,2]
            pass
        pass
    im_new = matrix_2_image(new_item) 
    return im_new

if __name__ == "__main__":
    file_name = "./4hx2.jpg"
    dst = smart_separate_ex(file_name)
    if dst != None:
        for item in dst:
            (name,im) = item
            im.save(name)



    
