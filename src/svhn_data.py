from scipy.io import loadmat
from numpy import array, moveaxis, append, shape

class ConvertFromMatFile():
    def __init__(self, dataPath):
        """ This class reads the input .mat-file provided in dataPath and converts it
        to a dictionary.
        This class also provides methods preprocessing the data to work with tensorflow.
        Calls `scipy.io.loadmat` with given path.
        Data from .mat-file will be saved internally.
        """
        self.__dataPath = dataPath
        self.__matData = loadmat(dataPath)

    def loadLabelsRaw(self):
        """
        ### Returns
        self.__matData['y'] : dict
                Unpreprocessed labels
        """
        return self.__matData['y']

    def loadLabels(self):
        """ Load the labels from dataset and correct them.
        Label data has the key 'y'.
        In the dataset '0' is labeled as '10'.
        This function changes every '10' back to '0'.
        ### Returns
        tmpLabels : list
                Corrected labels
        """
        tmpLabels = self.__matData['y']
        for i in range(0,len(tmpLabels)):
            if tmpLabels[i] == 10:
                tmpLabels[i] = 0
        return tmpLabels
    
    def loadImages(self):
        """ Load image data and move the axis by -1 to get it to a 
        tensorflow compatoble shape.
        Original shape: (32, 32, 3, 73257)
        After correction: (73257, 32, 32, 3)
        ### Returns
        result : ndarray 
              Corrected image data
        """
        return moveaxis(array(self.__matData['X']), -1, 0)
    
    def loadRescaledImages(self, scale_value=255.0):
        """ Calls `loadImages` and converts the image data from 0-scale_value to 0-1 range.
        (In case of SVHN dataset scale_value = 255).
        This function is CPU heavy. Tensorflow recomends using the 
        Rescaling layer in model creation to benefit from GPU acceleration, if active.
        ### Parameter
        scale_value : float
                used for rescaling image data to values between 0-1
        ### Returns
        result : NDArray
        """
        return self.loadImages() / scale_value
    
    def loadImagesRaw(self):
        """
        Image data is marked with key 'X'.
        ### Returns
        self.__matData['X'] : list
                unpreprocessed image data
        """
        return self.__matData['X']
    
    def getLabelMap(self):
        """Return a label map as an ndarray, which contains all possible 
        numerical labels (once) in ascending order.
        Calls `loadLabels` and sorts all elements.
        Iterates through each element and checks if label
        is in the list. If not append the list and finally return it.
        ### Returns
        labelMap : NDArray
                Array of all possible classes
        """
        tmpData = sorted(self.loadLabels())
        old_tmpLabel = tmpData[0]
        labelMap = tmpData[0]

        for i in range(0,len(tmpData)):
            tmpLabel = tmpData[i]
            if tmpLabel != old_tmpLabel:
                old_tmpLabel = tmpLabel
                labelMap = append(labelMap, tmpLabel)
        
        return labelMap

    def getImageShape(self):
        """Returns the shape of the images as a tuple of ints.
        In case of the SVHN dataset it will be (32, 32, 3).
        ### Returns
        shape : tuple[int, ...]
        """
        return shape(self.loadImages())[1:]
    
    def getDataPath(self):
        """
        ### Returns
        self.__dataPath : string
                data path of the loaded .mat-file
        """
        return self.__dataPath


    ###### Following methods were used for debugging. ######

    def __getDataRaw__(self):
        """
        Used for debugging.
        ### Returns
        data : dict
                contains the complete and unprocessed data from the .mat file
        """
        return self.__matData
    
    def __getKeys__(self):
        """
        Used for debugging.
        ### Returns
        keys : dict_keys
                dictionary keys of dataset
        """
        return self.__matData.keys()
    
    def __getHeader__(self):
        """ Information about dataset.
        Used for debugging.
        ### Returns
        header : string
        """
        return self.__matData["__header__"]
    
    def __getVersion__(self):
        """ Information about dataset.
        Used for debugging.
        ### Returns
        version : string
        """
        return self.__matData["__version__"]
    
    def getGlobals(self):
        """ Information about dataset.
        Used for debugging.
        ### Returns
        globals : string
        """
        return self.__matData["__globals__"]