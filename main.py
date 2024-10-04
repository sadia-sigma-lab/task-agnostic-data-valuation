import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
from tensorflow.keras.datasets import cifar10

# Set a random seed for reproducibility
np.random.seed(42)


# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images by dividing by 255 (pixel range 0-255)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the images into 2D arrays (samples, features)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


# Let's assume the buyer has a subset of classes 0-4
buyer_data = x_train[y_train.flatten() <= 4]

# Perform PCA on the buyer's data
pca = PCA(n_components=10)  # You can adjust the number of components
buyer_pca = pca.fit(buyer_data)

# Store the principal components and explained variances
buyer_components = buyer_pca.components_
buyer_variances = buyer_pca.explained_variance_

# Let's assume the seller has a subset of classes 5-9
seller_data = x_train[y_train.flatten() >= 5]

# Project the seller's data onto the buyer's principal components
seller_pca = pca.fit(seller_data)

# Get the seller's explained variances in the directions of the buyer's components
seller_variances = seller_pca.explained_variance_


# Calculate diversity and relevance
def calculate_diversity_relevance(buyer_variances, seller_variances):
    # Diversity: difference in variances
    diversity = np.sqrt(np.prod(np.abs(buyer_variances - seller_variances) / np.maximum(buyer_variances, seller_variances)))

    # Relevance: intersection of variances
    relevance = np.sqrt(np.prod(np.minimum(buyer_variances, seller_variances) / np.maximum(buyer_variances, seller_variances)))

    return diversity, relevance

# Compute diversity and relevance between buyer's and seller's data
diversity, relevance = calculate_diversity_relevance(buyer_variances, seller_variances)

print(f"Diversity: {diversity}, Relevance: {relevance}")



# Visualizing diversity vs. relevance
def plot_diversity_relevance(diversity, relevance, seller_description):
    plt.scatter(relevance, diversity, label=seller_description)
    plt.title("Diversity vs. Relevance")
    plt.xlabel("Relevance")
    plt.ylabel("Diversity")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_diversity_relevance(diversity, relevance, "Seller with Classes 5-9")

