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