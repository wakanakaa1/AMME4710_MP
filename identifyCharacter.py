# This file is used as the main file to identify the characters. 

import os 
import identifyCharacterFunction as functions

# Select a range of characters
bothCannonImage = 'Characters/bothCannon/360.png'
bothKnightImage = 'Characters/bothKnight/360.png'
bothRookImage = 'Characters/bothRook/360.png'
greenElephantImage = 'Characters/greenElephant/360.png'
greenKingImage = 'Characters/greenKing/360.png'
greenPawnImage = 'Characters/greenPawn/360.png'
redElephantImage = 'Characters/redElephant/360.png'
redGuardImage = 'Characters/redGaurd/360.png'
redKingImage = 'Characters/redKing/360.png'
redPawnImage = 'Characters/redPawn/1.png'

# The name of the folder each image is located in
folder = [
    (bothCannonImage, "Both Cannon"),
    (bothKnightImage, "Both Knight"),
    (bothRookImage, "Both Rook"),
    (greenElephantImage, "Green Elephant"),
    (greenKingImage, "Green King"),
    (greenPawnImage, "Green Pawn"),
    (redElephantImage, "Red Elephant"),
    (redGuardImage, "Red Guard"),
    (redKingImage, "Red King"),
    (redPawnImage, "Red Pawn")
]

print("Unpacking Classifer File")
modelFile = 'BestClassifier.pkl'
model, label = functions.loadSVM(modelFile)
print("Finished Unpacking")

for imagePath, folderName in folder:

    processedImage = functions.preProcessing(imagePath)
    HOGFeatures = functions.extractHOG(processedImage)

    topPredictions = functions.predictKLabel(model, HOGFeatures, label, k=3)

    print(f"\nImage: {os.path.basename(imagePath)}")
    print("Top predictions (ordered by confidance):")
    for scene, confidance in topPredictions:
        print(f"Scene: {scene}, Confidence: {confidance:.4f}")