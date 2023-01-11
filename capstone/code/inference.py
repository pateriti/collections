import streamlit as st
import io
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import yolov5


st.title('Shoe Image and Logo Detector')

file_name = st.file_uploader('Upload an image of a shoe')  

if file_name is not None:
    col1, col2 = st.columns(2)
    
    bytes_data = file_name.read() #read the BytesIO object in buffer
    test_image = image.load_img(io.BytesIO(bytes_data))

    #convert the image to a matrix of numbers to feed into model
    test_image = image.img_to_array(test_image) # 1st: convert loaded image to array
    
    #2nd: https://www.tensorflow.org/api_docs/python/tf/expand_dims 
    #(to add additional 4th dummy dimension for batch on top of height, width, channel for a color image, 
    #to meet Tensorflow's expected no. of dimensions for input image
    test_image = np.expand_dims(test_image, axis=0)
    
    #3rd: to pre-process inputs to be in the same format expected by MobileNetV2
    test_image = preprocess_input(test_image) 
    mobilenet_model = load_model('../assets/mobilenet_model.h5')
    result = mobilenet_model.predict(test_image) 
    brands = {'adidas': 0, 'converse': 1, 'nike': 2}
    result = [(i, np.max(result)) for i, j in brands.items() if j == np.argmax(result)]
    
    col1.subheader(f'Predicted brand: {str(result[0][0])}')
    col1.subheader(f'Probability: {str(round(result[0][1],4))}')
    
    #object detection
    weight_path = '../assets/'+'logov7.pt'
    model = yolov5.load(weight_path)    
       
    #set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # set image
    img = Image.open(io.BytesIO(bytes_data))
 
    # perform inference
    results = model(img)
    
    # inference with larger input size
    results = model(img, size=1280)
    
    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    
    col2.subheader('Detected brand logos with bounding boxes and confidence')
    col2.image(results.render())