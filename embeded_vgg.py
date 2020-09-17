#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 00:05:30 2020

"""
## for building DNN model 
from keras import Sequential 
from keras.layers import Dense
from keras import initializers 
from keras import optimizers
## train test split 
from sklearn.model_selection import train_test_split


## for data preprocessing 
import os 
import glob
import numpy as np
from PIL import Image

## for FeatureEmbemding 
from keras_vggface.vggface import VGGFace
from sklearn.preprocessing import MinMaxScaler


class PreprocessData:
    def load_label_data(self,input_dir):
        X = list();
        Y = list();

        for image_dir in os.listdir(input_dir):
            if (image_dir=='true'):
                label = 1
            else:
                label = 0
            
            image_paths = glob.glob(os.path.join(input_dir,image_dir,'*.jpg'))            
            
            for image_file in image_paths:
                image = Image.open(image_file)
                image = image.convert('RGB')
                image = np.asarray(image)
                X.append(image/255)
                Y.append(label)
                
        self.X = np.asarray(X);
        self.Y = np.asarray(Y);
        return self.X , self.Y

    def save(self,path):
        np.savez_compressed(path, X=self.X, Y=self.Y)
        
class FeatureEmbemding:
    """Use VGGFace model to do feature embemding""" 
    def __init__(self,X,Y):
        self.X = X;
        self.label = Y;
    
    def generate_features(self):
        base_model = VGGFace(model='vgg16')
        self.features_raw = base_model.predict(self.X);
        
    def feature_rescale(self):
        scaler = MinMaxScaler();
        self.feature = scaler.fit_transform(self.features_raw);
        
    def get_feature_label(self):
        return self.feature, self.label
    
    def save(self,path):
        np.savez_compressed(path, feature=self.feature, label=self.label)
    

class MyModel:
    def build_model(self,lr=0.001,decay = 0.01):
        self.model = Sequential();
        init = initializers.RandomNormal(mean=0, stddev=1, seed=None);
        self.model.add(Dense(1024, input_shape=(2622,), activation='relu',kernel_initializer= init));
        self.model.add(Dense(512, activation="relu"));
        self.model.add(Dense(256,activation="relu")); 
        self.model.add(Dense(128,activation="relu"));
        self.model.add(Dense(64,activation="relu"));
        self.model.add(Dense(32,activation="relu"));
        self.model.add(Dense(16,activation="relu"));
        self.model.add(Dense(1,activation="sigmoid"));
        # learning rate 0.001
        optimizer_adam = optimizers.Adam(lr=lr,decay = decay)
        self.model.compile(loss='binary_crossentropy', optimizer= optimizer_adam,  metrics = ["accuracy"])         
        return self.model



def main():
    label_data = PreprocessData();
    X, Y = label_data.load_label_data('/Face_of_Idou/imagesout')
    # save data as check point 
    label_data.save('/Face_of_Idou/imagesout_XY.npz')
    
    ## feature embemdding 
    FE = FeatureEmbemding(X,Y);
    FE.generate_features();
    FE.feature_rescale();
    # save as check point 
    FE.save('/Face_of_Idou/imagesout_feature_label.npz');
    
    features, labels = FE.get_feature_label();
    x_train,x_test,y_train,y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42)
   
    ## training 
    model = MyModel();
    model.build_model();
    hist = model.model.fit(x_train,y_train,validation_split=0.1,shuffle=True,epochs = 300)
    model.model.predict_class(x_test,y_test)
    
    
    
    
    

        
    
    
    
    
    
    
    

    
    
