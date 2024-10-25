# import libraries
import os, cv2, numpy as np, pickle

def extractHog(dataFolder):
    
    # Initalise list for features and labels
    featureArray = []
    featureLabel = []

    # Initalise for loop to iterate through all the folders
    for pieceFolder in os.listdir(dataFolder):

        # Identify the path to images
        piecePath = os.path.join(dataFolder, pieceFolder)
        if not os.path.isdir(piecePath):
            continue

        # Iterate through every image in the folder
        for imageName in os.listdir(piecePath):

            if imageName == ".DS_Store":
                continue

            # Identify the image path and read it
            imagePath = os.path.join(piecePath, imageName)
            image = cv2.imread(imagePath)

            # Resize Image
            size = (128, 128)
            resizedImage = cv2.resize(image, size)

            # Grayscale Image 
            grayScaleImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

            # Set HOG parameters
            winSize = (128, 128)
            blockSize = (16, 16)
            blockStride = (8, 8)
            cellSize = (8, 8)
            bins = 9

            # Initalise and compute HOG
            hogDescriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, bins)
            feature = hogDescriptor.compute(grayScaleImage).flatten()

            # Place into array 
            featureArray.append(feature)
            featureLabel.append(pieceFolder)

    return np.array(featureArray), np.array(featureLabel)

def saveModel(fileName, classifier, encoder):
    savedFile = open(fileName, 'wb')
    pickle.dump((classifier, encoder), savedFile)
    savedFile.close()






