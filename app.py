import streamlit as st 
from PIL import Image
from keras.models import Sequential, Model, load_model
from resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense
import constants
import cv2
from keras.preprocessing.image import load_img, img_to_array

def predict(image_path): 
    model = VGG16()
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    # return highest probability 
    label = label[0][0]
    return label 


def build_resnet_model():
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
    model.add(Dense(constants.NUM_CLASSES, activation = 'softmax'))
    model.layers[0].trainable = False
    return model

def test_resnet_model(test_image):
    model = build_resnet_model()
    model.load_weights(constants.CHECKPOINT_PATH)
    image = img_to_array(test_image)
    im = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    print(model.predict(im))
    yhat = model.predict(im)
#     label = decode_predictions(yhat)
#     label = label[0][0]
#     return label
    return yhat

def main():
    st.title("Chest X-ray Image Classification Project")
    uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.write(image)
        st.image(image, caption='Chest X-ray', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = test_resnet_model(image)
        st.write(label)
        # st.write('%s (%.2f%%)' % (label[1], label[2]*100))


if __name__ == "__main__":
    main() 

