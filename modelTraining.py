# import libraries
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
import pickle
import modelTrainingFunctions as function

# Set data folder
dataFolder = "Characters/"

# Load dataset and acquire HOG details
print('Starting HOG extraction')
featuresArray, featuresLabel = function.extractHog(dataFolder)
print('Finished extraction')

# Encode label class
encoder = LabelEncoder()
yEncoder = encoder.fit_transform(featuresLabel)

# Split data into training and testing sets (optional, if you want to test on a separate set)
X_train, X_test, y_train, y_test = train_test_split(featuresArray, yEncoder, test_size=0.2, random_state=42)

# Step 1: Define the parameter grid for the SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  # Gamma is relevant only for rbf kernel
}

# Step 2: Create the SVM model
svmClassifier = SVC(probability=True)

# Step 3: Initialize GridSearchCV with cross-validation
gridSearch = GridSearchCV(svmClassifier, param_grid, cv=5, verbose=2, n_jobs=-1)

# Step 4: Perform grid search on the training dataset
print("Starting Grid Search with cross-validation")
gridSearch.fit(X_train, y_train)

# Step 5: Print the best hyperparameters and the best score
print("Best hyperparameters found:", gridSearch.best_params_)
print("Best cross-validation score:", gridSearch.best_score_)

# Step 6: Train the final model using the best hyperparameters found
best_svm = gridSearch.best_estimator_
print("Training final model with best hyperparameters")
best_svm.fit(X_train, y_train)

# Step 7: Save the best model
modelFile = 'BestClassifier.pkl'
print("Saving the final model with best hyperparameters")
with open(modelFile, 'wb') as savedFile:
    pickle.dump((best_svm, encoder), savedFile)
print("Model saved successfully")

# Step 8: Make predictions on the test set
y_pred = best_svm.predict(X_test)

# Step 9: Calculate accuracy, precision, recall, F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' for multiclass
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Step 10: Calculate top-k accuracy (e.g., top-3 accuracy)
y_prob = best_svm.predict_proba(X_test)  # Get the predicted probabilities
top_k_acc = top_k_accuracy_score(y_test, y_prob, k=3)

print(f"Top-3 Accuracy: {top_k_acc:.4f}")