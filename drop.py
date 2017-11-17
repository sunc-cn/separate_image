#encoding:utf-8

import queue 

#用队列和集合记录遍历过的像素坐标代替单纯递归以解决cfs访问过深问题
def cfs(im,x_fd,y_fd):

    xaxis=[]
    yaxis=[]
    pix=im.load()
    visited =set()
    q=queue.Queue()
    q.put((x_fd, y_fd))
    visited.add((x_fd, y_fd))
    offsets=[(1, 0), (0, 1), (-1, 0), (0, -1)]#四邻域

    while not q.empty():
        x,y=q.get()

        for xoffset,yoffset in offsets:
            x_neighbor,y_neighbor=x+xoffset,y+yoffset

            if (x_neighbor,y_neighbor) in (visited):
                continue  # 已经访问过了

            visited.add((x_neighbor, y_neighbor))

            try:
                if pix[x_neighbor, y_neighbor]==0:
                    xaxis.append(x_neighbor)
                    yaxis.append(y_neighbor)
                    q.put((x_neighbor,y_neighbor))

            except IndexError:
                pass

    xmax=max(xaxis)
    xmin=min(xaxis)
    #ymin,ymax=sort(yaxis)

    return xmax,xmin

#搜索区块起点
def detectFgPix(im,xmax):
    l,s,r,x=im.getbbox()
    print(l,s,r,x)
    pix=im.load()
    for x_fd in range(xmax+1,r):
        for y_fd in range(x):
            if pix[x_fd,y_fd]==0:
                print(x_fd,y_fd)
                return x_fd,y_fd

def CFS(im):
    zoneL=[]#各区块长度L列表
    zoneBE=[]#各区块的[起始，终点]列表
    zoneBegins=[]#各区块起点列表
    xmax=0#上一区块结束黑点横坐标,这里是初始化
    for i in range(5):
        try:
            x_fd,y_fd=detectFgPix(im,xmax)
            print(x_fd,y_fd)
            xmax,xmin=cfs(im,x_fd,y_fd)
            L=xmax-xmin
            zoneL.append(L)
            zoneBE.append([xmin,xmax])
            zoneBegins.append(xmin)
        except TypeError:
            print("error")
            return zoneL,zoneBE,zoneBegins

    return zoneL,zoneBE,zoneBegins

#========>CFS ABOVE
#求出图片的垂直投影直方图
def VerticalProjection(im):

    VerticalProjection={}
    l,s,r,x=im.getbbox()

    pix=im.load()

    for x_ in range(r):

        black=0
        for y_ in range(x):

            if pix[x_,y_]==0:
                black+=1

        item=str(x_)
        VerticalProjection[item]=black

    return VerticalProjection

#========>VERTICALPROJECTION ABOVE
def zonexCutLines(zoneL,zoneBegins):

    Dmax= 23 #最大字符长度，人工统计后填入
    Dmin= 11 #最小字符长度,人工统计后填入
    Dmean= 16 #平均字符长度，人工统计后填入

    zonexCutLines=[]

    for i in range(len(zoneL)):

            xCutLines=[]     
            if zoneL[i]>Dmax:

                num=round(float(zoneL[i])/float(Dmean))
                num_int=int(num)

                if num_int==1:
                    continue

                for j in range(num_int-1):
                    xCutLine=zoneBegins[i]+Dmean*(j+1)
                    xCutLines.append(xCutLine)
                zonexCutLines.append(xCutLines)

            else:
                continue

    return zonexCutLines
	
def yVectors_sorted(zoneBE,VerticalProjection):

    yVectors_dict={}
    yVectors_sorted=[]
    for zoneBegin,zoneEnd in zoneBE:
        L=zoneEnd-zoneBegin
        Dmean= 16  #基于人工统计的平均字符长度值
        num=round(float(L)/float(Dmean))#区块长度L除以平均字符长度Dmean四舍五入可得本区块字符数量
        num_int=int(num)

        if num_int>1:#当本区块字符数量>1时候，可以认为出现字符粘连，是需要切割的区块

            for i in range(zoneBegin,zoneEnd+1):

                i=str(i)
                yVectors_dict[i]=VerticalProjection[i]#扣取需要切割的区块对应的垂直投影直方图的部分
            #对扣取部分进行重排并放入yVectors_sorted列表中   
            yVectors_sorted.append(sorted(yVectors_dict.items(),key=lambda d:d[1],reverse=False))

    return yVectors_sorted

def get_dropsPoints(zoneL,zonexCutLines,yVectors_sorted):

    Dmax= 23
    Dmean= 16
    drops=[]
    # yVectors_sorted__=[]
    # xCutLines=[]


    h=-1

    for j in range(len(zoneL)):
        yVectors_sorted_=[]

        if zoneL[j]>Dmax:

            num=round(float(zoneL[j])/float(Dmean))
            num_int=int(num)

            #容错处理
            if num_int==1:
                continue

            h+=1
            yVectors_sorted__=yVectors_sorted[h]
            xCutLines=zonexCutLines[h]

            #分离
            yVectors_sorted_x=[]
            yVectors_sorted_vector=[]
            for x,vector in yVectors_sorted__:
                yVectors_sorted_x.append(x)
                yVectors_sorted_vector.append(vector)

            for i in range(num_int-1):

                for x in yVectors_sorted_x:

                    x_int=int(x)
                    #d表示由Dmean得出的切割线和垂直投影距离的最小点之间的距离
                    d=abs(xCutLines[i]-x_int)

                    #d和Dmean一样也需要人工设置
                    if d<4:
                        drops.append(x_int)#x是str格式的 
                        break 

        else:

            #print '本区块只有一个字符'
            continue

    return drops
def get_Wi(im,Xi,Yi):

    pix=im.load()
    #statement
    n1=pix[Xi-1,Yi+1]
    n2=pix[Xi,Yi+1]
    n3=pix[Xi+1,Yi+1]
    n4=pix[Xi+1,Yi]
    n5=pix[Xi-1,Yi]

    if n1==255:
        n1=1
    if n2==255:
        n2=1
    if n3==255:
        n3=1
    if n4==255:
        n4=1
    if n5==255:
        n5=1

    S=5*n1+4*n2+3*n3+2*n4+n5

    if S==0 or S==15:
        Wi=4

    else:
        Wi=max(5*n1,4*n2,3*n3,2*n4,n5)

    return Wi

def situations(Xi,Yi,Wi):

    switcher={

        1: lambda:(Xi-1,Yi),
        2: lambda:(Xi+1,Yi),
        3: lambda:(Xi+1,Yi+1),
        4: lambda:(Xi,Yi+1),
        5: lambda:(Xi-1,Yi+1),
    }

    func=switcher.get(Wi,lambda:switcher[4]())
    return func()

#改进型就是在drops滴水起点的获取方式和经典不一样
def dropPath(im,drops):

    l,s,r,x=im.getbbox()    
    path=[]
    zonePath=[]    
    for drop in drops:

        Xi=drop
        Yi=0
        limit_left=drop-4#左约束
        limit_right=drop+4#右约束

        while Yi!=x-1:
            Wi=get_Wi(im,Xi,Yi)
            Xi,Yi=situations(Xi,Yi,Wi)

            if Xi==limit_left or Xi==limit_right:
                Xi,Yi=path[-1]#若触碰到约束边界，就回退到上一次的坐标

            if Yi>2:
                #如果遇到当前水滴位置坐标和上或者上上次的坐标一样，则设置为权重4，即垂直向下从n0挪动到n2的位置
                if path[-2]==(Xi,Yi) or path[-1]==(Xi,Yi):
                    Xi,Yi=situations(Xi,Yi,4)

            path.append((Xi,Yi))
        zonePath.append(path)
    return zonePath

#主函数
def DropCUT(im):
    pix=im.load()
    drops=Drops(im)
    zonePath=dropPath(im,drops)
    for path in zonePath:
        for x,y in path:
            pix[x,y]=255#令滴水路径上的所有坐标都染上白色

    return im
def get_Wi(im,Xi,Yi):

    pix=im.load()
    #statement
    n1=pix[Xi-1,Yi+1]
    n2=pix[Xi,Yi+1]
    n3=pix[Xi+1,Yi+1]
    n4=pix[Xi+1,Yi]
    n5=pix[Xi-1,Yi]

    if n1==255:
        n1=1
    if n2==255:
        n2=1
    if n3==255:
        n3=1
    if n4==255:
        n4=1
    if n5==255:
        n5=1

    S=5*n1+4*n2+3*n3+2*n4+n5

    if S==0 or S==15:
        Wi=4

    else:
        Wi=max(5*n1,4*n2,3*n3,2*n4,n5)

    return Wi

def situations(Xi,Yi,Wi):

    switcher={

        1: lambda:(Xi-1,Yi),
        2: lambda:(Xi+1,Yi),
        3: lambda:(Xi+1,Yi+1),
        4: lambda:(Xi,Yi+1),
        5: lambda:(Xi-1,Yi+1),
    }

    func=switcher.get(Wi,lambda:switcher[4]())
    return func()

#改进型就是在drops滴水起点的获取方式和经典不一样
def dropPath(im,drops):

    l,s,r,x=im.getbbox()    
    path=[]
    zonePath=[]    
    for drop in drops:

        Xi=drop
        Yi=0
        limit_left=drop-4#左约束
        limit_right=drop+4#右约束

        while Yi!=x-1:
            Wi=get_Wi(im,Xi,Yi)
            Xi,Yi=situations(Xi,Yi,Wi)

            if Xi==limit_left or Xi==limit_right:
                Xi,Yi=path[-1]#若触碰到约束边界，就回退到上一次的坐标

            if Yi>2:
                #如果遇到当前水滴位置坐标和上或者上上次的坐标一样，则设置为权重4，即垂直向下从n0挪动到n2的位置
                if path[-2]==(Xi,Yi) or path[-1]==(Xi,Yi):
                    Xi,Yi=situations(Xi,Yi,4)

            path.append((Xi,Yi))
        zonePath.append(path)
    return zonePath

#其实Ｄrops(im)作用很简单，就是把前面的求出水滴起始落点坐标的所有步骤全部串起来

def Drops(im):

    #PartONE
    zoneL,zoneBE,zoneBegins=CFS(im)
    print("partone",zoneL,zoneBE,zoneBegins)
    #PartTWO
    zonexCutLines_=zonexCutLines(zoneL,zoneBegins)

    #PartThree
    yVectors_sorted_=yVectors_sorted(zoneBE,VerticalProjection(im))

    #PartFOUR
    drops=get_dropsPoints(zoneL,zonexCutLines_,yVectors_sorted_)
    #print '最佳滴水点横坐标分别为为====>',drops

    return drops

#主函数
def DropCUT(im):

    pix=im.load()
    #print(pix[1,1])
    drops=Drops(im)
    zonePath=dropPath(im,drops)
    for path in zonePath:
        for x,y in path:
            pix[x,y]=255#令滴水路径上的所有坐标都染上白色

    return im



from PIL import Image  
import numpy as np
from PIL import ImageFilter  
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        #r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
def MatrixToImage(data):
    data = data
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

if __name__ == "__main__":
    raw_file_name = "./lqhr.jpg"
    raw_image = np.array(Image.open(raw_file_name))
    gray_image = convert2gray(raw_image)
    #print(gray_image)
    h, w = gray_image.shape
    im = gray_image
    new_item = np.zeros(shape=(32,90,3),dtype="uint8")
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
    im = MatrixToImage(new_item) 
    #im = im.filter(ImageFilter.MinFilter(3))  

    #im.save("gray.jpg")

    #im = Image.open(raw_file_name)
    im_grey = im.convert("L")
    im_peak = im_grey.convert("1")
    im_new = DropCUT(im_peak)
    im_new.save("new.jpg")
