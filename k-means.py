from collections import Counter
import numpy as np
import random
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


def normalize_images(images):
    images = np.array(images)
    normalized_images = images / 255.0
    return normalized_images


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def cosine_distance(x1, x2):
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def k_means(images, labels, distance, num_clusters=4, max_iterations=100):
    images = [np.array(img) for img in images]
    centroids = [images[i] for i in random.sample(range(len(images)), num_clusters)]
    centroids = [np.array(c) for c in centroids]
    cluster_labels = {i: [] for i in range(num_clusters)}
    sse = 0

    for iteration in range(max_iterations):
        clusters = {i: [] for i in range(num_clusters)}
        for i in range(num_clusters):
            cluster_labels[i] = []

        for image, label in zip(images, labels):
            if distance == "euclidean":
                distances = [
                    euclidean_distance(image, centroid) for centroid in centroids
                ]
            else:
                distances = [cosine_distance(image, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(image)
            cluster_labels[closest_centroid].append(label)
            sse += min(distances) ** 2

        new_centroids = []
        for cluster in clusters.values():
            if cluster:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(images[random.randint(0, len(images) - 1)])

        if all(
            np.array_equal(centroids[i], new_centroids[i]) for i in range(num_clusters)
        ):
            break

        centroids = new_centroids

    return clusters, centroids, sse, cluster_labels


def calculate_accuracy(cluster_labels):
    total_length = 0
    true_label_count = 0
    for i in cluster_labels.keys():
        cluster = cluster_labels.get(i)
        counter = Counter(cluster)
        total_length += len(cluster)
        true_label_count += counter.most_common(1)[0][1]
    accuracy = true_label_count / total_length
    print("Accuracy:", accuracy)


train_images = read_images("train-images.idx3-ubyte")
train_labels = read_labels("train-labels.idx1-ubyte")
correct_train_images, correct_train_labels = get_correct_digits(
    train_labels, train_images
)

# We checked if the normalization improves the accuracy, but it did not.
# normalized_train_images = normalize_images(correct_train_images)
normalized_train_images = correct_train_images

clusters, centroids, sse, cluster_labels = k_means(
    normalized_train_images, correct_train_labels, "euclidean", num_clusters=4
)

cluster_sizes = {k: len(v) for k, v in clusters.items()}
print("Euclidean Cluster sizes:", cluster_sizes)
print("Euclidean SSE:", sse)
calculate_accuracy(cluster_labels)


clusters, centroids, sse, cluster_labels = k_means(
    normalized_train_images, correct_train_labels, "cosine", num_clusters=4
)
cluster_sizes = {k: len(v) for k, v in clusters.items()}
print("Cosine Cluster sizes:", cluster_sizes)
print("Cosine SSE:", sse)
calculate_accuracy(cluster_labels)

# PCA
pca = PCA(n_components=50)
pca.fit(normalized_train_images)
transformed_images = pca.transform(normalized_train_images)

clusters, centroids, sse, cluster_labels = k_means(
    transformed_images, correct_train_labels, "euclidean", num_clusters=4
)
cluster_sizes = {k: len(v) for k, v in clusters.items()}
print("PCA Euclidean Cluster sizes:", cluster_sizes)

print("PCA SSE:", sse)
calculate_accuracy(cluster_labels)


clusters, centroids, sse, cluster_labels = k_means(
    transformed_images, correct_train_labels, "cosine", num_clusters=4
)
cluster_sizes = {k: len(v) for k, v in clusters.items()}
print("PCA Cosine Cluster sizes:", cluster_sizes)
print("PCA SSE:", sse)
calculate_accuracy(cluster_labels)
