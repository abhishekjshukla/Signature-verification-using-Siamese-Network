




"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))
"""







from keras import backend as K
from PIL import Image
import cv2
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge,Conv2D,MaxPool2D,ZeroPadding2D,Dense,Flatten,concatenate,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocess import normalize_image,remove_background






lst=['1','2','3','4']



real=[]
forg=[]
for i in range(3):
    real.append(os.listdir('Dataset/dataset'+lst[i]+'/real/'))
    forg.append(os.listdir('Dataset/dataset'+lst[i]+'/forge/'))


img=plt.imread('Dataset/dataset'+lst[0]+'/real/'+real[0][0])
plt.imshow(img)

def res(path,nm):
    mywidth = 500
    img = Image.open(path)
    wpercent = (mywidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
    img.save(nm)

for k in range(3):
    pathg='Dataset/dataset'+lst[k]+'/real/'
    pathf='Dataset/dataset'+lst[k]+'/forge/'
    for j in range(len(real[k])):
            path=pathg+real[k][j]
            res(path,path)
    for j in range(len(forg[k])):
            path=pathf+forg[k][j]
            res(path,path)

mxg=0
for k in range(3):
    pathg='Dataset/dataset'+lst[k]+'/real/'
    pathf='Dataset/dataset'+lst[k]+'/forge/'
    for j in range(len(real[k])):
            path=pathg+real[k][j]
            img=cv2.imread(path)
            if(img.shape[0]>mxg):
              a=img.shape
              mxg=img.shape[0]
    for j in range(len(forg[k])):
            path=pathf+forg[k][j]
            img=cv2.imread(path)
            if(img.shape[0]>mxg):
              a=img.shape
              mxg=img.shape[0]

for k in range(3):
    pathg='Dataset/dataset'+lst[k]+'/real/'
    pathf='Dataset/dataset'+lst[k]+'/forge/'
    for j in range(len(real[k])):
        path=pathg+real[k][j]
        img=cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgnr=normalize_image(gray,(mxg,500))
        cv2.imwrite(path,imgnr)
    for j in range(len(forg[k])):
        path=pathf+forg[k][j]
        img=cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgnr=normalize_image(gray,(mxg,500))
        cv2.imwrite(path,imgnr)   

xx2=np.zeros((3,549,500))

x=[]
xx=[]
m=0
for k in range(3):
    pathg='Dataset/dataset'+lst[k]+'/real/'
    pathf='Dataset/dataset'+lst[k]+'/forge/'
    for j in range(len(real[k])):
        for i in range(len(real[k])):
            if(real[k][j][-7:-4]==real[k][i][-7:-4] and real[k][i]!=real[k][j]):
                for n in range(len(real[k])):
                    if(real[k][i][-7:-4]==forg[k][n][-7:-4]):
                        anchor=cv2.imread(pathg+real[k][j])
                        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
                        positive=cv2.imread(pathg+real[k][i])
                        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2GRAY)
                        negative=cv2.imread(pathf+forg[k][n])
                        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)
                        x.append([anchor,positive,negative])
                        final=np.stack((anchor,positive,negative))
                        np.hstack((xx2,final))
                        xx.append([real[k][j],real[k][i],forg[k][n]])
x=np.asarray(x)

y=np.zeros((6000,1))

x_train,x_test,y_train,y_test=train_test_split(x.reshape(6000, 3, 549, 500,1),y,test_size=0.2,random_state=42)

anchor_tr=[]
positive_tr=[]
negative_tr=[]
for i in range(len(x_train)):
  anchor_tr.append(x_train[i][0])
  positive_tr.append(x_train[i][1])
  negative_tr.append(x_train[i][2])


anchor_ts=[]
positive_ts=[]
negative_ts=[]
for i in range(len(x_test)):
  anchor_ts.append(x_test[i][0])
  positive_ts.append(x_test[i][1])
  negative_ts.append(x_test[i][2])


anchor_tr=np.asarray(anchor_tr)
positive_tr=np.asarray(positive_tr)
negative_tr=np.asarray(negative_tr)


anchor_ts=np.asarray(anchor_ts)
positive_ts=np.asarray(positive_ts)
negative_ts=np.asarray(negative_ts)






# Architecture of network
def create_shared_network(shape):
  inp=Input(shape=shape)
  
  layer=Conv2D(96,kernel_size=(11,11),strides=(4,4),padding="same",activation="relu")(inp)
  layer=MaxPool2D(pool_size=(3,3),strides=(2,2))(layer)
  layer=ZeroPadding2D(padding=(2,2))(layer)  
  layer=Conv2D(256,kernel_size=(5,5),padding="valid",activation="relu")(layer)
  layer=MaxPool2D(pool_size=(3,3),strides=(2,2))(layer)
  
  layer=Conv2D(384,kernel_size=(3,3),padding="same",activation="relu")(layer)
  
  layer=Conv2D(384,kernel_size=(3,3),padding="same",activation="relu")(layer)
  
  
  layer=Conv2D(256,kernel_size=(3,3),padding="valid",activation="relu")(layer)
  layer=MaxPool2D(pool_size=(3,3),strides=(2,2))(layer)
  layer=Flatten()(layer)
  layer=Dense(1024,activation="relu")(layer)
  layer=Dense(1024,activation="relu")(layer)
  layer=Dense(256)(layer)
  out = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(layer)
  model=Model(inputs=inp,outputs=out)
  return model



md=create_shared_network((549,500,1))
md.summary()





ALPHA = 0.8
beta=128
epsilon=1e-8
N=128
def triplet_loss(x):
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss
  

def build_model(input_shape):
    anchor_example = Input(shape=input_shape)
    positive_example = Input(shape=input_shape)
    negative_example = Input(shape=input_shape)

    shared_network = create_shared_network(input_shape)

    anchor_shared = shared_network(anchor_example)
    positive_shared = shared_network(positive_example)
    negative_shared = shared_network(negative_example)
    
    loss = merge([anchor_shared, positive_shared, negative_shared],
                 mode=triplet_loss, output_shape=(1,))    
    
    model = Model(inputs=[anchor_example, positive_example, negative_example],
                  outputs=loss)
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mae", optimizer='adam')

    return model





model=build_model((549,500,1))

model.summary()

checkpoint = ModelCheckpoint('wt/model_2:'+str(k)+'  {epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.fit([anchor_tr,positive_tr,negative_tr], y_train, batch_size=100, epochs=50, verbose=1,callbacks=[checkpoint], validation_data=([anchor_ts,positive_ts,negative_ts],y_test)) 







