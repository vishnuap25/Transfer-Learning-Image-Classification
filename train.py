# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:09:42 2020

@author: VISHNU
"""

#Import libraries
import os
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf


#Sets the current working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)


class Image_Classifier:
  def __init__(self, model, img_size, epochs, batch_size, train_path, 
            validation_path, model_path, log_path):
    #Model Parameters
    self.IMAGE_SIZE = img_size
    self.BATCH_SIZE = batch_size
    self.EPOCHS = epochs
    self.MODEL = model
    #data path
    self.train_path = train_path
    self.valid_path = validation_path
    self.model_path = model_path
    self.log_path = log_path

    self.get_Output_Class()
    self.model_Initializer()
    self.Load_Data()
    self.run_Training()
  

  def model_Initializer(self):
    
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Model
    import tensorflow as tf

    #Resources
    print("Num GPUs Available: ", 
    len(tf.config.experimental.list_physical_devices('GPU')))
    print("Using Tensorflow : ",tf.__version__)

    # initializing the network model and excluding the last layer of network
    if self.MODEL == 'VGG16':
      from tensorflow.keras.applications.vgg16 import VGG16
      self.model = VGG16(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'VGG19':
      from tensorflow.keras.applications.vgg19 import VGG19
      self.model = VGG19(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'Xception':
      from tensorflow.keras.applications.xception import Xception
      self.model = Xception(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'ResNet50V2':
      from tensorflow.keras.applications.resnet_v2 import ResNet50V2
      self.model = ResNet50V2(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'ResNet101V2':
      from tensorflow.keras.applications.resnet_v2 import ResNet101V2
      self.model = ResNet101V2(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'ResNet152V2':
      from tensorflow.keras.applications.resnet_v2 import ResNet152V2
      self.model = ResNet152V2(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'InceptionV3':
      from tensorflow.keras.applications.inception_v3 import InceptionV3
      self.model = InceptionV3(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'InceptionResNetV2':
      from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
      self.model = InceptionResNetV2(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'MobileNetV2':
      from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
      self.model = MobileNetV2(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'DenseNet121':
      from tensorflow.keras.applications.densenet import DenseNet121
      self.model = DenseNet121(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'DenseNet169':
      from tensorflow.keras.applications.densenet import DenseNet169
      self.model = DenseNet169(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    if self.MODEL == 'DenseNet201':
      from tensorflow.keras.applications.densenet import DenseNet201
      self.model = DenseNet201(input_shape=self.IMAGE_SIZE + [3], 
            weights='imagenet',  #using pretrained imagenet weights
            include_top=False)   #excluding the last layer of network

    # Freezing the layes of the network
    for layer in self.model.layers:
      layer.trainable = False
    
    #flatterning the last layer
    self.x = Flatten()(self.model.output)

    #Created a dense layer for output
    self.outlayers = Dense(self.count_output_classes, activation='softmax')(self.x)

    #Binding vgg layers and custom output layer
    self.model = Model(inputs=self.model.input, 
              outputs=self.outlayers)

    #Compile the Model
    self.model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
      )


  def get_Output_Class(self):
    #gets output classes from the training folder
    self.output_classes = os.listdir(self.train_path)
    #number of output classes
    self.count_output_classes = len(self.output_classes)  


  
  def Load_Data(self):
    #Image Transformation Template
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    self.train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    self.test_datagen = ImageDataGenerator(rescale = 1./255)

    #Loading Image from Train directories
    self.training_set = self.train_datagen.flow_from_directory(self.train_path,
                                                  target_size = self.IMAGE_SIZE,
                                                  batch_size = self.BATCH_SIZE,
                                                  class_mode = 'categorical')

    #Loading Image from Test directories
    self.test_set = self.test_datagen.flow_from_directory(self.valid_path,
                                          target_size = self.IMAGE_SIZE,
                                          batch_size = self.BATCH_SIZE,
                                          class_mode = 'categorical')


  def run_Training(self):
    from tensorflow.keras.callbacks import TensorBoard
    self.tensorboard_callback = TensorBoard(log_dir=self.log_path,
                              update_freq="epoch")
    # Initiate Training 
    self.train = self.model.fit_generator(
                              self.training_set,
                              #validation_data=self.test_set,
                              epochs=self.EPOCHS,
                              steps_per_epoch=len(self.training_set),
                              #validation_steps=len(self.test_set),
                              callbacks=[self.tensorboard_callback]
                              )
    #save model for inference
    self.model.save(f'{self.model_path}/classification_model.h5')
  
if __name__ == "__main__": 
  parser = argparse.ArgumentParser(description='Collect model parameters')
  parser.add_argument('--model', default = 'VGG16', help="""pretrined model name.....Xception, VGG16, 
                      VGG19, ResNet50V2, ResNet101V2, ResNet152V2, InceptionV3, InceptionResNetV2, 
                      MobileNetV2, DenseNet121, DenseNet169, DenseNet201 default : VG16""")
  parser.add_argument('--img_size', default = [224, 224],type=list, help='Image size... default : [224, 224]')
  parser.add_argument('--epochs', default = 5,type=int, help='number of epochs... default : 5')
  parser.add_argument('--batch_size', default = 32,type=int, help='batch size for training...default : 32')
  parser.add_argument('--train_path', default = 'data/training_set', help='location of the training images.....default : data/training_set')
  parser.add_argument('--validation_path', default = 'data/test_set', help='location of the testing images.....default : data/test_set')
  parser.add_argument('--model_path', default = 'model/', help='location od model file to be saved.....default : model/')
  parser.add_argument('--log_path', default = 'logs/', help='location of log path.....default : logs/')
  args = parser.parse_args()
  classifier = Image_Classifier(args.model, args.img_size, args.epochs, args.batch_size,
                                args.train_path, args.validation_path, args.model_path, args.log_path)




















