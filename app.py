from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import OrderedDict
import tensorflow
from imutils import face_utils
import numpy as np
import imutils
import shutil
import dlib
import cv2
from flask import Flask, render_template, request, redirect, flash
import os

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask-app/assets', 
            template_folder='./flask-app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/webcam.html')
def webcam():
   return render_template('webcam.html')

@app.route('/about.html')
def about():
   return render_template('about.html')



@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   FACIAL_LANDMARKS_IDXS = OrderedDict([
       ("right_eye", (36, 42)),
       ("left_eye", (42, 48))
   ])
   detector = dlib.get_frontal_face_detector()
   predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
   image = cv2.imread("upload_chest.jpg")
   image = imutils.resize(image, width=500)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   rects = detector(gray, 1)
   for (i, rect) in enumerate(rects):
      shape = predictor(gray, rect)
      shape = face_utils.shape_to_np(shape)
      for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
         clone = image.copy()
         cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
         for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
         img_name = name + ".jpg"
         cv2.imwrite(img_name, roi)
   model= load_model('models/model.hdf5')
   left_url="left_eye.jpg"
   right_url="right_eye.jpg"
   shutil.copy2('left_eye.jpg', 'flask-app')
   shutil.copy2('right_eye.jpg', 'flask-app')
   left_image = tensorflow.keras.preprocessing.image.load_img('left_eye.jpg',target_size=(224, 224))
   right_image = tensorflow.keras.preprocessing.image.load_img('right_eye.jpg',target_size=(224, 224))
   left_image = tensorflow.keras.preprocessing.image.img_to_array(left_image)
   right_image = tensorflow.keras.preprocessing.image.img_to_array(right_image)
   left_image = np.expand_dims(left_image, axis=0)
   right_image = np.expand_dims(right_image, axis=0)
   left_image = left_image.reshape(1, 224, 224, 3)  
   right_image = right_image.reshape(1, 224, 224, 3) 

   pred_left = model.predict(left_image, batch_size=1)
   pred_right = model.predict(right_image, batch_size=1)
   probability_left = pred_left[0]
   probability_right = pred_right[0]
   if probability_left[0] > 0.7:
      model_pred_left = str('%.2f' % (probability_left[0]*100) + '% YES')
   else:
     model_pred_left = str('%.2f' % ((1-probability_left[0])*100) + '% NO')
   if probability_right[0] > 0.7:
       model_pred_right = str('%.2f' % (probability_right[0] * 100) + '% YES')
   else:
       model_pred_right = str('%.2f' % ((1 - probability_right[0]) * 100) + '% NO')
   return render_template('preditions_result.html',model_pred_left=model_pred_left,model_pred_right=model_pred_right, left_url=left_url,right_url=right_url)
if __name__ == '__main__':
   app.secret_key = ".."
   app.run()