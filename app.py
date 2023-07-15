# from flask import Flask,render_template,url_for, request, redirect
# import os
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import pickle
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow import keras

# app = Flask(__name__)

# # modelp = pickle.load(open('model.pkl', 'rb'))
# modelh = keras.models.load_model('modelpost.h5')

# def pred_label(test_apple_url):
#     class_names = ['freshapples', 'freshbanana', 'freshoranges',
#                'rottenapples', 'rottenbanana', 'rottenoranges']
#     img = tf.keras.utils.load_img(
#     test_apple_url, target_size=(100, 100))
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # create a batch

#     # predictions_apple = modelp.predict(img_array)
#     predictions_apple = modelh.predict(img_array)
#     score_apple = tf.nn.softmax(predictions_apple[0])

#     if(class_names[np.argmax(score_apple)][:6] == "rotten"):
#         print("This", class_names[np.argmax(score_apple)][6:], " is {:.2f}".format(
#         100-(100 * np.max(score_apple))), "% healthy")
#         formatted_number = format(100-(100 * np.max(score_apple)), ".2f")
#         return formatted_number,class_names[np.argmax(score_apple)]



#     else:
#         print("This", class_names[np.argmax(score_apple)][5:], " is {:.2f}".format(
#         100 * np.max(score_apple)), "% healthy")
#         formatted_number = format(100 * np.max(score_apple), ".2f")
#         return formatted_number,class_names[np.argmax(score_apple)]


# @app.route('/',methods=['GET','POST'])
# def home():
#     return render_template('index.html')

# @app.route('/submit',methods=['GET','POST'])
# def output():
#     if request.method=='POST':
#         img=request.files['fruitimage']
#         img_path="static/"+img.filename
#         img.save(img_path)
#         p=pred_label(img_path)
#         return render_template("index.html",prediction=p,img_path=img_path)
        
    


# if __name__ == "__main__":
#     app.run(debug=True)
#     # app.run(host="0.0.0.0",port=5000)







import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import streamlit as st
from PIL import Image
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
import time


# pickle_in=open('model.pkl','rb')
# loaded_model=pickle.load(pickle_in)
modelh = keras.models.load_model('modelpost.h5')
img_height= 100
img_width = 100
batch_size = 32




def fruit(test_apple_url):
    class_names = ['freshapples', 'freshbanana', 'freshoranges',
               'rottenapples', 'rottenbanana', 'rottenoranges']
    img = tf.keras.utils.load_img(
    test_apple_url, target_size=(100, 100))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  
    predictions_apple = modelh.predict(img_array)
    score_apple = tf.nn.softmax(predictions_apple[0])

    if(class_names[np.argmax(score_apple)][:6]=="rotten"):
        return ("This",class_names[np.argmax(score_apple)][6:]," is {:.2f}".format(100-(100 * np.max(score_apple))),"% healthy")
    else:
        return ("This",class_names[np.argmax(score_apple)][5:]," is {:.2f}".format(100 * np.max(score_apple)),"% healthy")


    
def main():
    add_bg_from_url() 
    st.title("Health of fruits")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:   
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Fruit Image")
        result = fruit(uploaded_file)
        st.header(f"{result[0]} {result[1]} {result[2]}%healthy")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://image.slidesdocs.com/responsive-images/background/cute-cartoon-with-fruits-and-flowers-powerpoint-background_bd0d8c9b12__960_540.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



if __name__== '__main__':
    main()
    # block is only executed when the script is run directly and not when it is imported as a module.
