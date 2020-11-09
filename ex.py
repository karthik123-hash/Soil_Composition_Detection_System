import PIL
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import os
from flask import Flask, flash, request, redirect, url_for
from flask import *  
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
'''
UPLOAD_FOLDER = 'C:/Users/happy family/Documents/bacup/soil'
path=("C:/Users/happy family/Documents/sudhi/sudheer/soilimg/151.jpg")
#img=cv2.imread("C:/Users/happy family/Documents/sudhi/sudheer/soilimg/151.jpg")
#plt.imshow(img)
image = Image.open(path)
#image.show()
left = 1500
top = 1000
right = 2500
bottom = 1600
crop_image=image.crop((left,top,right,bottom))
outpath="C:/Users/happy family/Documents/bacup/soil"
src_fname, ext = os.path.splitext(path) 
save_fname = os.path.join(outpath, os.path.basename(src_fname)+'.jpg')
crop_image.save(save_fname)

mean_rgb=[]
rgbimage=cv2.imread("C:/Users/happy family/Documents/bacup/soil/151.jpg")
for i in range(3):
    red_channel=rgbimage[:,:,i]
    mean_rgb.append((np.mean(red_channel)))
print(mean_rgb)
r,g,b=mean_rgb
test=[r,g,b]
print(test)
sn = open("soil_model.pkl","rb")
model = pickle.load(sn)
#test=[73.76331, 97.43942333333334 ,138.37434]
test=np.expand_dims(test,0)
print(test.shape)

prediction = model.predict(test)
print(prediction)
'''
#soildata=pd.read_csv('C:/Users/happy family/Documents/bacup/rgbmean.csv')
#import pandas as pd
a=['C1']
for i in a:
    x=i

soilclass=x
df=pd.read_csv('C:/Users/happy family/Documents/bacup/rgbmean.csv')
df=df[df['Category']==soilclass][['PH','EC','OC','P','K','S']]
data=df
#print(data)
dff=pd.read_csv('C:/Users/happy family/Documents/bacup/output_data.csv')
dff=dff[dff[ 'type']==soilclass][['details']]
print(dff)
aaa=dff.values.tolist()
#print(aaa)




