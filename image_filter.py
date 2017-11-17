from PIL import Image  
from PIL import ImageFilter  
  
def image_filters_test():  

    im = Image.open("./lqhr.jpg")  
  
    #预定义的图像增强滤波器  
    im_blur = im.filter(ImageFilter.BLUR)  
    im_blur.save("./blur.jpg")
    im_contour = im.filter(ImageFilter.CONTOUR)  
    im_contour.save("./contour.jpg")
    im_min = im.filter(ImageFilter.MinFilter(3))  
    im_min.save("./min.jpg")
    im_max= im.filter(ImageFilter.MaxFilter(3))  
    im_max.save("./max.jpg")
    im_min = im_contour.filter(ImageFilter.MinFilter(3))  
    im_min.save("./min2.jpg")
    
if __name__ == "__main__":
    image_filters_test()
