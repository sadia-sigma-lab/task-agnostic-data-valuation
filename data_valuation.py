import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import fashion_mnist

np.random.seed(42)

def preprocess_images(images, epsilon=1e-11):
    images = images.reshape(len(images), -1)  # Flatten the images
    images = images.astype('float32') / 255.0  # Normalize pixel values to [0,1]
    mean = np.mean(images, axis=0)
    std_dev = np.std(images, axis=0)

    # Avoid division by zero by adding a small epsilon to std_dev
    images = (images - mean) / (std_dev + epsilon)  # Zero-center and normalize
    return images

def get_data(images, labels, classes, exclude_indices=set(), num_samples=10000):
    indices = np.where(np.isin(labels, classes))[0]
    # Exclude already used indices to ensure distinct images
    indices = np.setdiff1d(indices, list(exclude_indices))
    np.random.shuffle(indices)
    selected_indices = indices[:num_samples]
    selected_images = images[selected_indices]
    selected_labels = labels[selected_indices]
    return selected_images, selected_labels, set(selected_indices)

def compute_eigenvectors_from_covariance(data):
    # Compute the covariance matrix for seller's data
    covariance_matrix = np.cov(data, rowvar=False)
    # Perform Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors

def compute_variance_in_sellers_direction(buyer_data, seller_eigenvectors):
    # Compute the covariance matrix for the buyer's data
    covariance_matrix = np.cov(buyer_data, rowvar=False)
    
    projected_eigenvalues = []
    # For each seller's eigenvector, compute the variance in buyer's data direction
    for i in range(seller_eigenvectors.shape[1]):
        u_i = seller_eigenvectors[:, i]  # Get the i-th seller eigenvector
        projected_cov = np.dot(u_i.T, np.dot(covariance_matrix, u_i))  # Project buyer's data on seller's eigenvector
        projected_eigenvalues.append(projected_cov)
    
    return np.array(projected_eigenvalues)

def compute_diversity_relevance(buyer_eigenvalues, seller_eigenvalues):
    max_vals = np.maximum(buyer_eigenvalues, seller_eigenvalues)
    min_vals = np.minimum(buyer_eigenvalues, seller_eigenvalues)
    abs_diff = np.abs(buyer_eigenvalues - seller_eigenvalues)

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        diversity_components = abs_diff / max_vals
        relevance_components = min_vals / max_vals

    # Replace NaN values with zero
    valid_mask = ~np.isnan(diversity_components) & ~np.isnan(relevance_components)

    # Apply the mask to filter out rows containing NaN in either array
    diversity_components = diversity_components[valid_mask]
    relevance_components = relevance_components[valid_mask]

    diversity = np.prod(diversity_components)
    diversity = diversity ** (1 / len(diversity_components))
    relevance = np.prod(relevance_components)
    relevance = relevance ** (1 / len(relevance_components))
    return diversity, relevance

def load_fashion_mnist_data():
    # Load fashion-MNIST dataset
    (fmnist_train_images, fmnist_train_labels), (fmnist_test_images, fmnist_test_labels) = fashion_mnist.load_data()
    fmnist_images = np.concatenate((fmnist_train_images, fmnist_test_images))
    fmnist_labels = np.concatenate((fmnist_train_labels, fmnist_test_labels))
    return fmnist_images, fmnist_labels

def prepare_buyer_data(fmnist_images, fmnist_labels):
    buyer_classes = [0, 1, 2, 3, 4]
    buyer_images, buyer_labels, buyer_indices = get_data(
        fmnist_images, fmnist_labels, buyer_classes, num_samples=6000)
    return buyer_images, buyer_labels, buyer_indices

def prepare_sellers_data(fmnist_images, fmnist_labels, used_indices):
    sellers_info = [
        ("Seller 1 (Classes 0-4)", [0, 1, 2, 3, 4]),
        ("Seller 2 (Classes 1-5)", [1, 2, 3, 4, 5]),
        ("Seller 3 (Classes 0-9)", list(range(0, 10))),
        ("Seller 4 (Classes 3-9)", [3, 4, 5, 6, 7, 8, 9]),
        ("Seller 5 (Classes 5-9)", [5, 6, 7, 8, 9])
    ]

    seller_images_list = []
    seller_labels_list = []
    seller_names = []
    for name, classes in sellers_info:
        images, labels, indices = get_data(
            fmnist_images, fmnist_labels, classes, exclude_indices=used_indices, num_samples=10000)
        used_indices.update(indices)
        seller_images_list.append(images)
        seller_labels_list.append(labels)
        unique_labels = np.unique(labels)
        print("Labels for {}: {}".format(name, unique_labels))
        seller_names.append(name)

    # Sellers from fashion-MNIST
    # Seller 6: Sandal (Class 5)
    seller6_indices = np.where(fmnist_labels == 5)[0]
    seller6_indices = np.setdiff1d(seller6_indices, list(used_indices))
    np.random.shuffle(seller6_indices)
    seller6_images = fmnist_images[seller6_indices[:10000]]
    seller6_labels = fmnist_labels[seller6_indices[:10000]]
    used_indices.update(seller6_indices[:10000])
    seller_images_list.append(seller6_images)
    seller_labels_list.append(seller6_labels)
    seller_names.append("Seller 6 (Sandal)")

    # Seller 7: Coat (Class 4)
    seller7_indices = np.where(fmnist_labels == 4)[0]
    seller7_indices = np.setdiff1d(seller7_indices, list(used_indices))
    np.random.shuffle(seller7_indices)
    seller7_images = fmnist_images[seller7_indices[:10000]]
    seller7_labels = fmnist_labels[seller7_indices[:10000]]
    used_indices.update(seller7_indices[:10000])
    seller_images_list.append(seller7_images)
    seller_labels_list.append(seller7_labels)
    seller_names.append("Seller 7 (Coat)")

    # Seller 8: Noisy images
    seller8_images = np.random.normal(0, 1, (10000, 28, 28))
    seller8_labels = None  # No labels
    seller_images_list.append(seller8_images)
    seller_labels_list.append(seller8_labels)
    seller_names.append("Seller 8 (Noise)")

    return seller_images_list, seller_labels_list, seller_names, used_indices

def compute_diversity_and_relevance_for_sellers(buyer_images, seller_eigenvector_list , seller_eigenvalues_list):
    diversity_vals = []
    relevance_vals = []
    # Preprocess Buyer's Data
    buyer_data = preprocess_images(buyer_images)
    
    for index, seller_eigenvectors in enumerate(seller_eigenvector_list):
        # Compute Buyer's variance along each Seller's Eigenvectors
        buyer_eigenvalues = compute_variance_in_sellers_direction(buyer_data, seller_eigenvectors)
        # print("buyer_eigen_values shape" , buyer_eigenvalues.shape)
        # print("seller_eigen_values shape" , seller_eigenvalues_list[0].shape)
        # Compute Diversity and Relevance
        diversity, relevance = compute_diversity_relevance(buyer_eigenvalues, seller_eigenvalues_list[index])
        diversity_vals.append(diversity)
        relevance_vals.append(relevance)
    return diversity_vals, relevance_vals

def plot_diversity_vs_relevance(seller_names, diversity_vals, relevance_vals):
    # Plot Diversity vs. Relevance
    plt.figure(figsize=(10, 8))
    plt.scatter(relevance_vals, diversity_vals, color='blue', s=100)
    for i, txt in enumerate(seller_names):
        plt.annotate(txt, (relevance_vals[i], diversity_vals[i]), fontsize=12)
    plt.title("Diversity vs. Relevance of Seller Data", fontsize=16)
    plt.xlabel("Relevance", fontsize=14)
    plt.ylabel("Diversity", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))  # X-axis grid interval of 0.1
    plt.yticks(np.arange(0, 1.1, 0.1))  # Y-axis grid interval of 0.1

    # Adding grid
    plt.grid(True)
    plt.show()
