#encoding:utf-8

import Queue

#用队列和集合记录遍历过的像素坐标代替单纯递归以解决cfs访问过深问题
def cfs(im,x_fd,y_fd):

    xaxis=[]
    yaxis=[]
    pix=im.load()
    visited =set()
    q=Queue.Queue()
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
    pix=im.load()
    for x_fd in range(xmax+1,r):
        for y_fd in range(x):
            if pix[x_fd,y_fd]==0:
                return x_fd,y_fd

def CFS(im):

    zoneL=[]#各区块长度L列表
    zoneBE=[]#各区块的[起始，终点]列表
    zoneBegins=[]#各区块起点列表

    xmax=0#上一区块结束黑点横坐标,这里是初始化
    for i in range(5):

        try:

            x_fd,y_fd=detectFgPix(im,xmax)
            xmax,xmin=cfs(im,x_fd,y_fd)
            L=xmax-xmin
            zoneL.append(L)
            zoneBE.append([xmin,xmax])
            zoneBegins.append(xmin)

        except TypeError:
            return zoneL,zoneBE,zoneBegins

    return zoneL,zoneBE,zoneBegins

#========>CFS ABOVE