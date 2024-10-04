import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import fashion_mnist
import torch
from scipy.linalg import null_space

# Load necessary data and define helper functions
def preprocess_images(images, epsilon=1e-11):
    images = images.reshape(len(images), -1)
    images = images.astype('float32') / 255.0
    mean = np.mean(images, axis=0)
    std_dev = np.std(images, axis=0)
    return (images - mean) / (std_dev + epsilon)

def get_data(images, labels, classes, exclude_indices=set(), num_samples=10000):
    indices = np.where(np.isin(labels, classes))[0]
    indices = np.setdiff1d(indices, list(exclude_indices))
    np.random.shuffle(indices)
    selected_indices = indices[:num_samples]
    selected_images = images[selected_indices]
    selected_labels = labels[selected_indices]
    return selected_images, selected_labels, set(selected_indices)

def compute_projected_eigenvalues(cov_matrix, components, num_components):
    projected_eigenvalues = []
    for i in range(num_components):
        u_i = components[i]
        projected_cov = np.dot(u_i.T, np.dot(cov_matrix, u_i))
        projected_eigenvalues.append(projected_cov)
    return np.array(projected_eigenvalues)

def compute_diversity_relevance(buyer_eigenvalues, seller_eigenvalues):
    max_vals = np.maximum(buyer_eigenvalues, seller_eigenvalues)
    min_vals = np.minimum(buyer_eigenvalues, seller_eigenvalues)
    abs_diff = np.abs(buyer_eigenvalues - seller_eigenvalues)

    with np.errstate(divide='ignore', invalid='ignore'):
        diversity_components = abs_diff / max_vals
        relevance_components = min_vals / max_vals

    valid_mask = ~np.isnan(diversity_components) & ~np.isnan(relevance_components)
    diversity_components = diversity_components[valid_mask]
    relevance_components = relevance_components[valid_mask]

    diversity = np.prod(diversity_components) ** (1 / len(diversity_components))
    relevance = np.prod(relevance_components) ** (1 / len(relevance_components))
    return diversity, relevance

def load_fashion_mnist_data():
    (fmnist_train_images, fmnist_train_labels), (fmnist_test_images, fmnist_test_labels) = fashion_mnist.load_data()
    fmnist_images = np.concatenate((fmnist_train_images, fmnist_test_images))
    fmnist_labels = np.concatenate((fmnist_train_labels, fmnist_test_labels))
    return fmnist_images, fmnist_labels

def EMCov(X, args, b_budget=False, b_fleig=True):
    rho = args['total_budget']
    delta = args['delta']
    n = args['n']
    d = args['d']
    cov = torch.mm(X.t(), X)
    if not(delta > 0.0):
        eps_total = np.sqrt(2 * rho)
    else:   
        eps_total = rho + 2. * np.sqrt(rho * np.log(1 / delta))
    eps0 = 0.5 * eps_total
    Uc, Dc, Vc = torch.svd(cov)
    lap = torch.distributions.laplace.Laplace(0, 2. / eps0).sample((d,))
    Lamb_hat = torch.diag(Dc) + torch.diag(lap)
    Lamb_round = torch.zeros(d)
    if b_fleig:
        for i in range(d):
            lamb = max(min(Lamb_hat[i, i], n), 0)
            Lamb_round[i] = lamb
    P1 = torch.eye(d)
    Ci = cov
    Pi = torch.eye(d)
    theta = torch.zeros(d, d)
    for i in range(d):
        Ci, Pi = EMStep(cov, Ci, Pi, eps0 / d, d, i, theta)
    C_hat = torch.zeros(d, d)
    for i in range(d):
        C_hat = C_hat + Lamb_round[i] * torch.outer(theta[i], theta[i])
    return C_hat / n

def EMStep(C, Ci, Pi, epi, d, i, theta):
    # Perform SVD and ensure index does not exceed the matrix dimensions
    U, S, V = torch.svd(Ci)
    
    if i < V.shape[1]:  # Ensure i is within the bounds
        u_hat = V[:, i]
    else:
        u_hat = V[:, -1]  # Use the last column if i exceeds the dimension
    
    theta_hat = torch.matmul(Pi, u_hat)
    theta[i] = theta_hat
    
    if not (i == d - 1):
        Pi = torch.from_numpy(null_space(theta[:i+1].numpy()))  # Recalculate Pi based on reduced theta
        Ci = torch.matmul(torch.matmul(Pi.t(), C), Pi)
    
    return Ci, Pi


def preprocess_and_pca(buyer_images):
    buyer_data = preprocess_images(buyer_images)
    pca = PCA()
    pca.fit(buyer_data)
    buyer_eigenvalues = pca.explained_variance_
    buyer_components = pca.components_
    significant_components = buyer_eigenvalues > 0.04
    return buyer_eigenvalues[significant_components], buyer_components[significant_components]

def prepare_buyer_data(fmnist_images, fmnist_labels):
    buyer_classes = [0, 1, 2, 3, 4]
    return get_data(fmnist_images, fmnist_labels, buyer_classes, num_samples=6000)

def prepare_sellers_data(fmnist_images, fmnist_labels, used_indices):
    sellers_info = [("Seller 1 (Classes 0-4)", [0, 1, 2, 3, 4]), ("Seller 2 (Classes 1-5)", [1, 2, 3, 4, 5])]
    seller_images_list = []
    seller_labels_list = []
    seller_names = []
    for name, classes in sellers_info:
        images, labels, indices = get_data(fmnist_images, fmnist_labels, classes, exclude_indices=used_indices, num_samples=10000)
        used_indices.update(indices)
        seller_images_list.append(images)
        seller_labels_list.append(labels)
        seller_names.append(name)
    return seller_images_list, seller_labels_list, seller_names, used_indices

def compute_diversity_and_relevance_for_sellers(buyer_eigenvalues, buyer_components, seller_images_list, use_private_covariance=False, privacy_args=None):
    diversity_vals = []
    relevance_vals = []
    for seller_images in seller_images_list:
        seller_data = preprocess_images(seller_images)
        if use_private_covariance:
            seller_data_torch = torch.tensor(seller_data)
            seller_cov = EMCov(seller_data_torch, privacy_args)
        else:
            seller_cov = np.cov(seller_data, rowvar=False)
        seller_eigenvalues = compute_projected_eigenvalues(seller_cov, buyer_components, len(buyer_eigenvalues))
        diversity, relevance = compute_diversity_relevance(buyer_eigenvalues, seller_eigenvalues)
        diversity_vals.append(diversity)
        relevance_vals.append(relevance)
    return diversity_vals, relevance_vals

def plot_diversity_vs_relevance(seller_names, diversity_vals, relevance_vals, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(relevance_vals, diversity_vals, color='blue', s=100)
    for i, txt in enumerate(seller_names):
        plt.annotate(txt, (relevance_vals[i], diversity_vals[i]), fontsize=12)
    plt.title(title, fontsize=16)
    plt.xlabel("Relevance", fontsize=14)
    plt.ylabel("Diversity", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def main():
    fmnist_images, fmnist_labels = load_fashion_mnist_data()

    # Prepare buyer's data
    buyer_images, buyer_labels, buyer_indices = prepare_buyer_data(fmnist_images, fmnist_labels)
    used_indices = buyer_indices.copy()

    # Prepare sellers' data
    seller_images_list, seller_labels_list, seller_names, used_indices = prepare_sellers_data(fmnist_images, fmnist_labels, used_indices)

    # Preprocess buyer's data and perform PCA
    buyer_eigenvalues, buyer_components = preprocess_and_pca(buyer_images)

    # Privacy settings
    privacy_args = {'total_budget': 1.0, 'delta': 1e-5, 'n': 10000, 'd': 28 * 28}

    # Compute without privacy
    diversity_vals_non_priv, relevance_vals_non_priv = compute_diversity_and_relevance_for_sellers(buyer_eigenvalues, buyer_components, seller_images_list, use_private_covariance=False)

    # Compute with privacy
    diversity_vals_priv, relevance_vals_priv = compute_diversity_and_relevance_for_sellers(buyer_eigenvalues, buyer_components, seller_images_list, use_private_covariance=True, privacy_args=privacy_args)

    # Plot non-private diversity vs relevance
    plot_diversity_vs_relevance(seller_names, diversity_vals_non_priv, relevance_vals_non_priv, "Non-Private Diversity vs Relevance")

    # Plot private diversity vs relevance
    plot_diversity_vs_relevance(seller_names, diversity_vals_priv, relevance_vals_priv, "Private Diversity vs Relevance")

    # Plot difference between private and non-private results
    diff_diversity = np.abs(np.array(diversity_vals_priv) - np.array(diversity_vals_non_priv))
    diff_relevance = np.abs(np.array(relevance_vals_priv) - np.array(relevance_vals_non_priv))
    plt.figure(figsize=(10, 8))
    plt.scatter(diff_relevance, diff_diversity, color='red', s=100)
    plt.title("Difference in Diversity and Relevance (Private vs Non-Private)", fontsize=16)
    plt.xlabel("Difference in Relevance", fontsize=14)
    plt.ylabel("Difference in Diversity", fontsize=14)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
