import datetime
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA


def read_images(filename):
    with open(filename, "rb") as file:
        _, num_images, rows, cols = (
            int.from_bytes(file.read(4), "big"),
            int.from_bytes(file.read(4), "big"),
            int.from_bytes(file.read(4), "big"),
            int.from_bytes(file.read(4), "big"),
        )
        images = []
        for _ in range(num_images):
            image = []
            for __ in range(rows * cols):
                pixel = int.from_bytes(file.read(1), "big")
                image.append(pixel)
            images.append(image)
        return images


def read_labels(filename):
    with open(filename, "rb") as file:
        _, num_labels = int.from_bytes(file.read(4), "big"), int.from_bytes(
            file.read(4), "big"
        )
        labels = []
        for _ in range(num_labels):
            label = int.from_bytes(file.read(1), "big")
            labels.append(label)

        return labels


def get_correct_digits(labels, images, correct_digits=[2, 3, 8, 9]):
    result_labels = []
    result_images = []
    for index in range(len(labels)):
        if labels[index] in correct_digits:
            result_labels.append(labels[index])
            result_images.append(images[index])
    return result_images, result_labels


train_images = read_images("train-images.idx3-ubyte")
train_labels = read_labels("train-labels.idx1-ubyte")
test_images = read_images("t10k-images.idx3-ubyte")
test_labels = read_labels("t10k-labels.idx1-ubyte")


correct_train_images, correct_train_labels = get_correct_digits(
    train_labels, train_images
)
correct_test_images, correct_test_labels = get_correct_digits(test_labels, test_images)

X_train = np.array(correct_train_images).reshape(len(correct_train_images), -1)
y_train = np.array(correct_train_labels)
X_test = np.array(correct_test_images).reshape(len(correct_test_images), -1)
y_test = np.array(correct_test_labels)


pipeline = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42))

param_grid = {
    "linearsvc__C": [0.01, 0.1, 1, 10, 100],
    "linearsvc__loss": ["hinge", "squared_hinge"],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Current time is ", datetime.datetime.now())

# Question B-PCA
pca_components = 50
model_with_pca = make_pipeline(
    StandardScaler(),
    PCA(n_components=pca_components),
    LinearSVC(dual=False, C=1.0, random_state=42),
)

model_with_pca.fit(X_train, y_train)

pca = model_with_pca.named_steps["pca"]
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

y_train_pred_pca = model_with_pca.predict(X_train)
y_test_pred_pca = model_with_pca.predict(X_test)
# Question D
train_accuracy_pca = accuracy_score(y_train, y_train_pred_pca)
test_accuracy_pca = accuracy_score(y_test, y_test_pred_pca)

print(f"Training Accuracy with PCA: {train_accuracy_pca*100}")
print(f"Test Accuracy with PCA: {test_accuracy_pca*100}")
print(f"Current time is ", datetime.datetime.now())

model_with_rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))

param_grid = {
    "svc__C": [1, 10],
    "svc__gamma": ["scale", 1],
}

grid_search = GridSearchCV(model_with_rbf, param_grid, cv=3, scoring="accuracy")

grid_search.fit(X_train, y_train)

best_model_rbf = grid_search.best_estimator_

y_train_pred_rbf = best_model_rbf.predict(X_train)
y_test_pred_rbf = best_model_rbf.predict(X_test)

train_accuracy_rbf = accuracy_score(y_train, y_train_pred_rbf)
test_accuracy_rbf = accuracy_score(y_test, y_test_pred_rbf)

print(f"Training Accuracy with RBF kernel: {train_accuracy_rbf*100:.2f}%")
print(f"Test Accuracy with RBF kernel: {test_accuracy_rbf*100:.2f}%")
print("Best parameters:", grid_search.best_params_)

# Question D-PCA
grid_search_extracted = GridSearchCV(
    model_with_rbf, param_grid, cv=3, scoring="accuracy"
)

grid_search_extracted.fit(X_train_pca, y_train)

best_model_rbf_extracted = grid_search_extracted.best_estimator_

y_train_pred_rbf_extracted = best_model_rbf_extracted.predict(X_train_pca)
y_test_pred_rbf_extracted = best_model_rbf_extracted.predict(X_test_pca)

train_accuracy_rbf_extracted = accuracy_score(y_train, y_train_pred_rbf_extracted)
test_accuracy_rbf_extracted = accuracy_score(y_test, y_test_pred_rbf_extracted)

print(
    f"Training Accuracy with RBF kernel (Extracted Features): {train_accuracy_rbf_extracted*100:.2f}%"
)
print(
    f"Test Accuracy with RBF kernel (Extracted Features): {test_accuracy_rbf_extracted*100:.2f}%"
)
print("Best parameters:", grid_search_extracted.best_params_)
