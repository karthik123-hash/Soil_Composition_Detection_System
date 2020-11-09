import os
import PIL
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from flask import Flask, flash, request, redirect, url_for,render_template,session
from werkzeug.utils import secure_filename 
import pickle
global soilclass
global df
global list_values

UPLOAD_FOLDER = 'C:/Users/Karthik/Desktop/bacup/static/soil'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower()in ALLOWED_EXTENSIONS

def rgb_extract(full_filename):
    #print(full_filename)
    im = Image.open(full_filename)
    image = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    left = 1500
    top = 1000
    right = 2500
    bottom = 1600
    crop_image=image.crop((left,top,right,bottom))
    outpath="C:/Users/Karthik/Desktop/bacup/static/soil"
    src_fname, ext = os.path.splitext(full_filename) 
    save_fname = os.path.join(outpath, os.path.basename(src_fname)+'.jpg')
    crop_image.save(save_fname)
    mean_rgb=[]
    rgbimage=cv2.imread(full_filename)
    for i in range(3):
        red_channel=rgbimage[:,:,i]
        mean_rgb.append((np.mean(red_channel)))
    r,g,b=mean_rgb
    return r,g,b

def predict_class(r,g,b):

    sn = open("soil_model.pkl","rb")
    model = pickle.load(sn)
    test=[r,g,b]
    test=np.expand_dims(test,0)
    #print(test.shape)

    prediction = model.predict(test)
    #print(prediction)
    for i in prediction:
        x=i
    return x

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/result', methods = ['POST'])  
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
     
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            r,g,b=rgb_extract(full_filename)
            #print(r,g,b)
            global soilclass
            soilclass = predict_class(r,g,b)
            print(soilclass)
            # print(type(soilclass))
            # print("im here")
            global df
            df=pd.read_csv('C:/Users/Karthik/Desktop/bacup/rgbmean.csv')
            df=df[df['Category']==soilclass][['PH','EC','OC','P','K','S']]
            return render_template("result.html",result=soilclass,data=df.to_html(index=False))


@app.route("/result")
def goback():
    global df
    global soilclass
    return render_template("result.html",result=soilclass,data=df.to_html(index=False))        


@app.route('/fertility', methods = ['POST'])  
def check_fertility():

    dff=pd.read_csv('C:/Users/Karthik/Desktop/bacup/output_data.csv')
    global soilclass
    print("im in check",soilclass)
    dff=dff[dff[ 'type']==soilclass][['details']]
    #print(soilclass)
    global list_values
    list_values=dff.values.tolist()
    return render_template("fertility.html",outputData=list_values[0])

if __name__ == '__main__':  
    app.debug =True
    app.run()           