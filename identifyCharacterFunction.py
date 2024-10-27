# This file is used to store the functions that will be used in the identifications of the processed images. 
# Import libraries
import pickle, cv2, numpy as np

def loadSVM(data):
    file = open(data, 'rb')
    model, label = pickle.load(file)
    file.close()

    return model, label

def predictKLabel(SVMModel, HOGFeatures, label, k=3):

    # Predict class probabilities
    classProbability = SVMModel.predict_proba(HOGFeatures)[0]

    # Get top-K indecies sorted by its confidance values
    topKIndex = np.argsort(classProbability)[::-1][:k]

    # Retrieve class labels and probability
    topKLabel = label.inverse_transform(topKIndex)
    topKConfidance = classProbability[topKIndex]

    return list(zip(topKLabel, topKConfidance))

def preProcessing(imagePath):

    # Read image into function
    image = cv2.imread(imagePath)

    # Resize and grayscale
    size = (128, 128)
    resizedImage = cv2.resize(image, size)
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

    return grayImage

def extractHOG(grayImage):

    # Set parameters
    winSize = (128, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    bins = 9

    # Create HOG descriptor and features
    HOGDescriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, bins)
    features = HOGDescriptor.compute(grayImage)

    # Flatten and reshape array
    features = features.flatten().reshape(1, -1)

    return features
