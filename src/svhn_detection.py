from os import getcwd
from numpy import max, argmax
from random import randint
from matplotlib import pyplot as plt
import svhn_data as sd
import tensorflow as tf
from datetime import datetime as dt


class SVHN_Detection():
      def __init__(self):
            """ Set default values for data paths and variables.
            """
            self.file_directory = getcwd()
            self.data_path = ""
            self.train_path = ""
            self.validation_path = ""

            self.model_changed = False
            self.in_training = False
            self.epochs = 5

            self.fit_callbacks = [TrainingCallback(),
                                  tf.keras.callbacks.ProgbarLogger(stateful_metrics="accuracy",
                                                                   count_mode ="steps")
                                                                   ]

      def setDataset(self, path):
            """ 
            Accepts one or two file paths and checks if filename(s) contain "train" or "test".
            Loads the datasets respectively ("train" -> train dataset; "test" -> validation dataset).
            ### Parmeter
            path : (tuple[str, ...] | Literal[str])
                        contains the full path to the selected file(s)
            ### Returns
            train_data_loaded, test_data_loaded : tuple[bool, bool]
                        returns `True` if data is train and/or test data is loaded without 
                        raising an exception
            ### Raises
            - `NotImplementedError` 
                        When more than two files are selected.
            - `FileNotFoundError` 
                        When filename does not contain "train" or "test"
            ### Catches
            - any exception
                        when trying to load data from .mat-file
                        (Does no exception handling)
            """
            train_data_loaded = False
            test_data_loaded = False
            if len(path) > 2:
                  raise NotImplementedError("Only two files can be selected, containing the train amd test data.")
            for p in path:
                  if "train" in p:
                        self.train_path = p
                  elif "test" in p:
                        self.validation_path = p
                  else:
                        raise FileNotFoundError("File name does not contain 'train' or 'test'.")
            
            try:
                  self.__loadTrainData(self.train_path)
            except Exception as e:
                  pass
            else:
                  train_data_loaded = True
            
            try:
                  self.__loadValData(self.validation_path)
            except Exception as e:
                  pass
            else:
                  test_data_loaded = True
            
            return train_data_loaded, test_data_loaded
                  
      def setNumEpochs(self, epochs):
            """ Sets `self.epochs` to given parameter.
            ### Parameter
            epochs : int
                        number of epochs for training
            """
            self.epochs = epochs

      def getModelChanged(self):
            """ 
            ### Returns
            model_changed : bool
                        `True` if maodel has changed
            """
            return self.model_changed

      def getModel(self):
            """ 
            ### Returns
            model : Sequential
                        returns the tensorflow model
            """
            return self.model
           
      def createModel(self):
            """
            Sets `model_changed` to `True`.
            Initiate tensorflow keras model by adding layers.
            Each layer is given a name for further processing.
            - `RandomRotation` (`name` = "rand_rotation") : 
                        Rotates the image randomly up to the specified value.
            - `RandomZoom` (`name` = "rand_zoom") : 
                        Zooms randomly in (positive values) or out (negative values).
            - `Rescaling` (`name` = "rescale") : 
                        Images usually have values between 0-255 for each pixel. This layer 
            normalizes then to a range from 0-1 to match the desired values by tensorflow.
            - `Conv2D` (`name` = "conv_1", "conv_2") : 
                        Convolutes 2-dimensional input data.
            - `MaxPooling` (`name` = "pool_1", "pool_2") : 
                        Finds the largest values in a specified pool of pixel data.
            - `Dropout` (`name` = "dropout") : Randomly sets outputs to 0.
            - `Flatten` (`name` = "flatten") : Create a 1-dimensional output from a multi-dimensional input (here: 2D).
            - `Dense` (`name` = "dense_1", "dense_output") : Creates specified number of outputs.
            ### Returns
            model : Sequential
                        tensorflow model containing the layers
            """
            self.model_changed = True
            self.model = tf.keras.Sequential([
                  tf.keras.layers.RandomRotation(0.1, fill_mode="nearest", name="rand_rotation"),
                  tf.keras.layers.RandomZoom((-0.1,0.1), fill_mode="nearest", name="rand_zoom"),
                  tf.keras.layers.Rescaling(1./255, name="rescale"),
                  tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=self.image_shape, name="conv_1"),
                  tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), name="pool_1"),
                  tf.keras.layers.Conv2D(16, 3, activation='relu', name="conv_2"),
                  tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool_2"),
                  tf.keras.layers.Dropout(0.1, name="dropout"),
                  tf.keras.layers.Flatten(input_shape=self.image_shape, name="flatten"),
                  tf.keras.layers.Dense(128, activation='relu', name="dense_1"),
                  tf.keras.layers.Dense(self.num_classes, activation='softmax', name="dense_output")   
            ])
            return self.model

      def compileModel(self):
            """ Calls tensorflow's `complie` function with
            - `optimizer` = "adamax"
            - `loss` = tf.keras.losses.SparseCategoricalCrossentropy()
            - `metrics` = ["accuracy"]
            """
            self.model.compile(optimizer="adamax", 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                              metrics=["accuracy"])

      def fitModel(self, num_epochs=5):
            """ Trains the model with `train_images` and `train_labels`.
            20% of that data is used as validation data.
            Other parameters:
            - `epochs` = num_epochs
            - `validation_split` = 0.2
            - `shuffle` = True
            - `callbacks` = self.fit_callbacks
            ### Parameter
            num_epochs : int
                        specify number of epochs to train
            ### Returns
            fit : Any
                        used to display training history.
            """
            self.model_changed = True
            self.fit = self.model.fit(self.train_images, 
                                      self.train_labels, 
                                      epochs=num_epochs,
                                      validation_split=0.2,
                                      shuffle=True,
                                      callbacks=self.fit_callbacks)
            return self.fit
      
      def evaluateLoss(self, print_to_console=False):
            """
            Calls tensorflow's `evaluate` method using the validation dataset.
            ### Parameter
            print_to_console : bool (default = `False`)
                        specify whether to print loss and accuracy
            ### Returns
            loss, accuracy : tuple[float, float]
                        Calculated loss and accuracy with data the model has never seen.
            """
            images = self.validation_images
            labels = self.validation_labels
            
            loss, accuracy = self.model.evaluate(images, labels, verbose=2)

            if print_to_console:
                  print("Accuracy: ", accuracy)
                  print("Loss: ", loss)

            return loss, accuracy
 
      def plotLossAndAcc(self, fit, show_plot=True):
            """
            From [Tensorflow tutorials](https://www.tensorflow.org/tutorials/images/classification#visualize_training_results)\\
            Sets up the necessary environment for `matplotlib`. 
            Plots the graphs for accuracy and loss from training and training validation 
            over each epoch.
            ### Parameters
            1. fit : Any
                      - the model's data collected during training (`model.fit`)
            2. show_plot : bool (default = `False`)
                      - specify whether `plt.show()` should be called from this method. 
                        Useful to show multiple plots at once.
            """
            acc = fit.history["accuracy"]
            val_acc = fit.history["val_accuracy"]

            loss = fit.history["loss"]
            val_loss = fit.history["val_loss"]

            epochs_range = range(self.epochs)

            plt.figure(facecolor="#000000", layout="constrained")
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label="Training Accuracy")
            plt.plot(epochs_range, val_acc, label="Validation Accuracy")
            plt.grid()
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.title('Training Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label="Training Loss")
            plt.plot(epochs_range, val_loss, label="Validation Loss")
            plt.grid()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.title('Training Loss')

            if show_plot:
                  plt.show()

      def predictModel(self, images):
            """Predicts given input data and returns the predicted value.
            Calls tensorflow's `predict`.
            ### Parameter
            images : Any
                        image data to predict with trained model
            ### Returns
            prediction : Any
                        classification conaining the probabilities for all classes in each
            """
            prediction = self.model.predict(images)
            return prediction

      def saveModel(self, model_path):
            """ Save the trained model to a given path.
            If path matches the current working directory, create new directory "Saved_Models"
            and save model to "my_model_[current date and time]".
            Calls tensorflow's `.save` method.
            Resets `self.model_changed`.
            ### Raises 
            - `NotADirectoryError` if input string is empty.
            Other exceptions will be returned, if raised while saving the model.
            ### Catches 
            - any exception raised during model saving. 
            ### Returns
            e : (Exception | None)
                        Returns error message
            """
            if model_path == self.file_directory:
                  model_path = "/Saved_Models/my_model_{}".format(dt.now().strftime("%Y-%m-%d-%H-%M-%S"))
            elif model_path == None or model_path == "":
                  raise NotADirectoryError
            try:
                  self.model.save(model_path)
            except Exception as e:
                  return e
            else:
                  self.model_changed = False

      def loadModelFromSave(self, file_path, compile=False):
            """ Loads a previously saved model.
            Calls `tf.keras.models.load_model` and stores the result in `self.model`.
            Evaluates the model if `compile`is true and prints the results.
            Resets `self.model_changed`.
            ### Parameters
            1. file_path : string
                        path to directory of saved model
            2. compile : bool (default =`False`)
                        specify whether the model should be compiled when loadning
            ### Returns
            model : Sequential
                        new loaded model
            """
            print("Loading model...")
            self.model = tf.keras.models.load_model(file_path, compile=compile)
            if compile:
                  print("Evaluating...")
                  self.evaluateLoss(print_to_console=True)
            print("Model loaded.")
            self.model_changed = False
            return self.model
      
      def __plotImage(self, i, prediction_array, true_label, img):
            """ Plots one single image from given parameters.
            Selects the chosen image and label from `ìmg` and `true_label`.
            Picks the greatest value from `prediction_array` as the predicted label.
            If this predicted label is equal to the true label, colorize the x-label green, otherwise red.
            Add a label to each image conaining the predicted label from `self.class_names`, 
            the prediction probability in percent, and the true label.
            ### Parameters
            1. i : int
                  - number of image to plot
            2. prediction_array : Any 
                  - array containing class predictions from one image
            3. true_label : Any
                  - the correct labels of given images
            4. img : Any
                  - all images that were predicted
            """
            true_label, img = true_label[i], img[i]

            plt.imshow(img)
            
            predicted_label = argmax(prediction_array)
            
            if predicted_label == true_label:
                  color = 'green'
            else:
                  color = 'red'
            
            plt.xlabel("{} {:.0%} ({})".format(self.class_names[predicted_label],
                                                max(prediction_array),
                                                true_label),
                        color=color)

      def plotImageGrid(self, prediction, show_plot=True):
            """ Plots 25 random images with their predicted and true labels.
            Calls `__plotImage` for each image.
            ### Parameters
            1. prediction : Any
                  - array of predictions
            2. show_plot : bool (default = `True)`
                  - specifies whether plot (plt.show()) should be held back. 
                  Useful if more than one plot should be displayed at the same time.
            """
            plt.figure(facecolor="#000000", layout="constrained") 

            for i in range(1,26):
                  r = randint(1,len(self.validation_labels))
                  plt.subplot(5,5,i)
                  self.__plotImage(r, prediction[r], self.validation_labels, self.validation_images)
            
            if show_plot:
                  plt.show()

      def createAndTrain(self):
            """ Calls all neccessary methods for training and evaluation:
            - createModel()
            - compileModel()
            - fitModel(epochs)
            """
            self.createModel()
            self.compileModel()
            self.fitModel(self.epochs)
      
      def __loadTrainData(self, path):
            """ Sets all variables needed in training.
            Creates new instance to class `ConvertFromMatFile` and sets up train variables.
            Calls `loadImages` and `loadLabels`. Also calls `getLabelMap` and `getImageShape` 
            to adjust variables to dataset.
            This method should be called whenever the dataset changes.
            ### Parameter
            path : string
                        contains the path to the .mat file
            """
            self.train_data = sd.ConvertFromMatFile(path)
            self.train_images = self.train_data.loadImages()
            self.train_labels = self.train_data.loadLabels()

            self.class_names = self.train_data.getLabelMap()
            self.num_classes = len(self.class_names)
            self.image_shape = self.train_data.getImageShape()

      def __loadValData(self, path):
            """ Sets all variables needed for validation.
            Creates new instance to class `ConvertFromMatFile` and sets up validation variables.
            Calls `loadImages` and `loadLabels`. 
            This method should be called whenever the dataset changes.
            ### Parameter
            path : string
                        contains the path to the .mat file
            """
            self.validation_data = sd.ConvertFromMatFile(path)
            self.validation_images = self.validation_data.loadImages()
            self.validation_labels = self.validation_data.loadLabels()

      
class TrainingCallback(tf.keras.callbacks.Callback):
      """ Custom callback for model training.
      Sets a global flag to indicate when model is in training.
      """
      def on_train_begin(self, logs=None):
            """ When training starts set global variable `in_training` to `True`.
            """
            global in_training
            in_training = True

      def on_train_end(self, logs=None):
            """ When training starts set global variable `in_training` to `False`.
            """
            global in_training
            in_training = False