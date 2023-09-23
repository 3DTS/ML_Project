# Machine Learning GUI for SVHN dataset
GUI for training an ML-algorithm (TensorFlow) to classify digits of the SVHN-dataset. 
Since the SVHN-dataset is in a .mat-format any other image dataset in this format could be used for training, but it is not tested yet.
For this the .mat-file will be converted to a python dictionary with the image data labelled as "X" and labels as "y". Those names are hardcoded in svhn_data.py.

## Features
- Training a machine learning algorithm locally using data from a MatLab's .mat-files
- Save and load models
- Evaluation of accuracy and loss in a graph for each epoch (This only works when the model was trained in the current session. **Loaded models won't provide that info!**)
- (It *should* be independent of image dimensions. [**not tested!**])

## Start and usage
To start this program just execute gui.py (assumed all the necessary packages are installed).
The GUI should look like this (without the red markings):

![GUI screenshot](https://github.com/3DTS/ML_Project/assets/104661402/cc7a087a-235c-42b6-b93d-45ff26fbe5f1 "GUI screenshot")

No. | Description
---:|:---
**1**| Menu bar <br> - Files: offers functions such as loading and saving the model<br> - Help: Information about the program version, author, and GitHub repository
**2**| - Choose files: Loading the dataset<br> - Start training: start the training<br> - Number of Epochs: Define the number of iterations in training (values 0 - 50)
**3**| - Load model: Loading a previously saved model<br> - Save model: Save the currently loaded/trained model
**4**| - Evaluate: Calculate "loss" and "accuracy" with the "test" data set<br> - Images: Opens a new window with 5x5 randomly selected images of the "test" dataset<br> - Graph: Output of the history of "accuracy" and "loss" during training (with validation)
**5**| Console window: This is where all the outputs of the program are displayed.

### Train the model
1. Load files for training and test
    - The file containing the training data must contain "training" in the filename. Likewise, the file containing the test data must contain "test" in the filename.
2. Choose number of epochs (Usually more epochs lead to better results. But be sure not to [overfit](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/) the model!)
3. Hit "Start training"-button
4. Depending on various factors such as the used hardware and the number of epochs the training my take a while.
5. Pressing on "Evaluate" will test the model with the test-dataset, which is unknown to the model and the accuracy and loss will be printed to the console.
6. Save the model by pressing "Save model".
7. The button "Images" will evaluate the model (like in 5.) and show 25 random images with the predicted and true labels and the probability of that prediction.
8. "Graph" shows the course of accuracy and loss during each epoch of training. This will only work when a model was trained in the current session. (loaded models will not work)

## Structure
The program consists of three files: gui.py, svhn_detection.py and svhn_data.py. They contain the following classes:
-	gui.py:  
    -	`SVHN_GUI`
    -	`Console`
-	svhn_detection.py:
    -	`SVHN_Detection`
    -	`TrainingCallbacks`
-	svhn_data.py:
    -	`ConvertFromMatFile`
    -	`TrainingCallback` 

![simplified class model](https://github.com/3DTS/ML_Project/assets/104661402/b00747d3-8fb5-4402-8565-3e045d99a80f "simplified class model")

The `TrainingCallback` class is used in SVHN_Detection as a custom callback in training, and therefore has no relationship to the other classes shown here. It sets the global variable `in_training` to `True` 
when the model is trained. Otherwise, it is `False`. It should be noted that this function has not been used and therefore its functionality is not guaranteed.
 
### gui.py
From the interface, the user can make settings or start processes, which are then connected to `SVHN_Detection` and further processed. Therefore, it only includes the `SVHN_Detection` class of the project files. 
The GUI also offers the possibility to specify which settings may be made. This eliminates potential sources of error due to incorrect configurations.
All methods of the `SVHN_GUI` class are marked with two underscores because they act as internal methods and should not be called from the outside. 
Only the `Console` class provides methods for use from outside. This class implements the ability to redirect standard output from the console to a text box. In order for the functions of `sys.stdout` to be 
compatible with the `tkinter` widget, they must be redefined. So, if `sys.stdout.write` is called, the method from the `Console` class is called. This takes the content to be output and executes the `insert` method, 
which displays the content on the text box. Similarly, the other methods work. The `flush` method serves as a dummy here, because TensorFlow uses this method and otherwise an `AttributeError` is thrown. 
(Note: The class `Console` was taken from the Reddit user ["rdbende"]([https://www.reddit.com/r/Tkinter/comments/nmx0ir/how_to_show_terminal_output_in_gui/](https://www.reddit.com/r/Tkinter/comments/nmx0ir/comment/gzrq86t/?utm_source=share&utm_medium=web2x&context=3)https://www.reddit.com/r/Tkinter/comments/nmx0ir/comment/gzrq86t/?utm_source=share&utm_medium=web2x&context=3).)

### svhn_detection.py
The class `SVHN_Detection` provides all the methods and procedures required for the training. It imports the class `ConvertFromMatFile` (see svhn_data.py) to load the record. Most of the methods here are accessible from the outside,
as it is supposed to work like an API with other interfaces. They are primarily used to make settings and call TensorFlow functions. 
The TensorFlow model has the following structure:

```
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
```

### svhn_data.py
The `ConvertFromMatFile` class contains all the necessary methods for data preparation. With the help of the `loadmat` method of `scipy.io`, .mat-files can be read. 
When the class is instantiated, this method is called directly and the data is stored in an internal variable. For further processing, image and label data can be extracted from it.

## Helpful Links
SVHN Dataset: http://ufldl.stanford.edu/housenumbers/

TensorFlow's tutorial on image classification: https://www.tensorflow.org/tutorials/images/classification

Over- and Underfitting: https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/
