#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 00:38:04 2020
_alignment function reformated: 
https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py
"""
from PIL import Image
import glob
import face_recognition
import cv2
import numpy as np
import sys
import os
import time
import logging
#import multiprocessing as mp


class image_preprocess:
    def find_align_crop_image(self, input_path, output_path):
      image = face_recognition.load_image_file(input_path,)      
      self.face_locations = face_recognition.face_locations(image, model="cnn")
      self.face_landmarks = face_recognition.face_landmarks(image)
      
      if(len(self.face_locations) >1):
          print('More than one face in this image so ignore the image.')
          return
      else:
          try:
              self.output=self._alignment(image, self.face_landmarks)
              pil_image = Image.fromarray(self.output)
              pil_image.save(output_path)
          except:
              print('Unexprected error:',sys.exc_info()[0]);
       
    def _alignment(self, image, face_landmarks):
        leftEyePts = face_landmarks[0]['left_eye']
        rightEyePts = face_landmarks[0]['right_eye']
        leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
        rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")

        # calculat the rotation angle
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # set the ideal left and right eye location
        desiredLeftEye=(0.35, 0.45)
        desiredRightEyeX = 1.0 - desiredLeftEye[0]
        
        # set the croped image(face) size after rotaion.
        desiredFaceWidth = 224
        desiredFaceHeight = 224
        
        # calculate the desired the scale after crop 
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist
        
        # calculate the center of eyes
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) //2.0,(leftEyeCenter[1] + rightEyeCenter[1]) // 2.0)
        
        # calculat the rotation matrix
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)
        return output
    
    def input_and_output(self,input_dir,output_dir):
        start_time = time.time()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        #mp.log_to_stderr
        #logger =mp.get_logger()
        #logger.setLevel(logging.INFO)       
        #pool=mp.Pool(os.cpu_count())  
        
        for image_dir in os.listdir(input_dir):
            image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)
    
            image_paths = glob.glob(os.path.join(input_dir,image_dir,'*.jpg'))
            
            for index, image_file in enumerate(image_paths):
                image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_file)))                
                output_file = os.path.join(image_output_dir,str(index)+'.jpg')    
                self.find_align_crop_image(image_file, output_file)          
                #pool.apply_async(self.find_align_crop_image,(image_file, output_file))
                
        #pool.close()
        #pool.join()
        end_time = time.time()
        total_time = end_time-start_time;
        logging.info('Completed in {} seconds'.format(total_time))
        return
            
if __name__=='__main__':
    test = image_preprocess()
    test.input_and_output('./images/','./imagesout/')



