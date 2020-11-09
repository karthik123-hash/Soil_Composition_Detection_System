import cv2
import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import glob
class A():
    def __init__(self):
        self._data=pd.DataFrame([[0,0,0]],columns=['r','g','b'])
        img_dir = "C:/Users/sudheer/Documents/final/soilimg" # Enter Directory of all images 
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path)
        outpath='C:/Users/sudheer/Documents/final/cropedimage'
        for f1 in files:
            im = Image.open(f1)
            image = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            width,height=image.size
            left = 1500
            top = 1000
            right = 2500
            bottom = 1600
            crop_image=image.crop((left,top,right,bottom))
            #crop_image.save("C:/Users/sudheer/Documents/final/cropedimage/crp.jpeg","Jpeg")
            src_fname, ext = os.path.splitext(f1)  # split filename and extension
            # construct output filename, basename to remove input directory
            save_fname = os.path.join(outpath, os.path.basename(src_fname)+'.JPG')
            crop_image.save(save_fname)
            #data.append(img)
    def rgb_value(self):
        img_dir2 = "C:/Users/sudheer/Documents/final/cropedimage" # Enter Directory of all images 
        data_path = os.path.join(img_dir2,'*g')
        files2 = glob.glob(data_path)
        mean_rgb=[]
        list1=[]
        for f2 in files2:
            image1=cv2.imread(f2)
            cv2.waitKey(0)
            for i in range(3):
                red_channel=image1[:,:,i]
                mean_rgb.append((np.mean(red_channel)))
            r,g,b=mean_rgb
            mean_rgb.clear()
            rgb = np.array([r, g ,b])
            rgb_list = []
            #x = np.array([0, 0, 0])
            rgb_list.append(rgb)
            #rgb_list.append(x)
            rgb_list = np.asarray(rgb_list)
            data_temp=pd.DataFrame(rgb_list, columns = ["r","g","b"])
            self._data=pd.concat([self._data,data_temp])
            
    def write_rgb(self):
        os.chdir('C:/Users/sudheer/Documents/final')
        self._data.to_csv('rgbmean.csv',index=False)
            


def main():
    a=A()
    a.rgb_value()
    a.write_rgb()

if __name__== "__main__":
    main()
