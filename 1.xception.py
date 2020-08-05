from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import tensorflow as tf
from keras.models import load_model
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input

class xception:
    def __init__(self,train_path,model_save_path,test_image_path):
        self.train_path=train_path
        self.model_save_path=model_save_path
        self.test_image_path=test_image_path
    def train(self):
        # re-size all the images to this
        IMAGE_SIZE = [224,224]
        # add preprocessing layer to the front of VGG
        xception= Xception(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        # don't train existing weights
        for layer in xception.layers:
            layer.trainable = False
        # useful for getting number of classes
        folders = glob(self.train_path+'*')
        # our layers - you can add more if you want
        x = Flatten()(xception.output)
        prediction = Dense(len(folders), activation='sigmoid')(x)
        # create a model object
        model = Model(inputs=xception.input, outputs=prediction)
        # view the structure of the model
        model.summary()
        # tell the model what cost and optimization method to use
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        # Use the Image Data Generator to import the images from the dataset
        train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory(self.train_path,target_size = (224, 224),batch_size = 32,class_mode = 'categorical')
        test_set = test_datagen.flow_from_directory(self.train_path,target_size = (224, 224),batch_size = 32,class_mode = 'categorical')
        # fit the model
        r=model.fit_generator(training_set,validation_data=test_set,epochs=10,steps_per_epoch=2,validation_steps=len(test_set))
        model.save(self.model_save_path)
    def test(self):
        model = load_model(self.model_save_path)
        image = load_img(self.test_image_path, target_size=(224, 224))
        image1 = img_to_array(image)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        image1 = preprocess_input(image1)
        yhat = model.predict(image1)
        pred = np.argmax(yhat)
        #change according to your dataset
        if pred== 0:
            prediction = 'satellite_image'
            print(prediction)
            plt.imshow(image)
        else:
            prediction = 'non_satellite_image'
            print(prediction)
            plt.imshow(image)
if __name__=="__main__":
    train_path=input('Enter the Input folder path:')
    model_save_path=input('Enter the path to save model in terms of .h5')
    test_image_path=input('Enter the image path to test:')
    a=Xception(train_path,model_save_path,test_image_path)
    b=a.train()
    c=a.test() 
