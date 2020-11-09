from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

sn = open("soil_model.pkl","rb")
model = pickle.load(sn)
test=[73.76331, 97.43942333333334 ,138.37434]
test=np.expand_dims(test,0)
print(test.shape)

prediction = model.predict(test)
print(prediction)
for i in prediction:
    if i== 'C2':
        print("good soil")
    else:
        print("not healthy soil")

