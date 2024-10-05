from data_valuation import load_fashion_mnist_data, prepare_buyer_data, prepare_sellers_data, preprocess_images, compute_diversity_and_relevance_for_sellers , compute_eigenvectors_from_covariance , plot_diversity_vs_relevance
import numpy as np
def main():
    # Load data
    print("Hold on , Running........")
    fmnist_images, fmnist_labels = load_fashion_mnist_data()

    # Prepare Buyer's Data
    buyer_images, buyer_labels, buyer_indices = prepare_buyer_data(fmnist_images, fmnist_labels)
    used_indices = buyer_indices.copy()

    # Prepare Sellers' Data
    seller_images_list, seller_labels_list, seller_names, used_indices = prepare_sellers_data(
        fmnist_images, fmnist_labels, used_indices)

    # Threshold for eigenvalue filtering
    eigenvalue_threshold = 10**-1

    # Sellers compute their eigenvectors from their covariance matrices
    seller_eigenvector_list = []
    seller_eigenvalues_list = []
    for seller_images in seller_images_list:
        seller_data = preprocess_images(seller_images)
        seller_eigenvalues, seller_eigenvectors = compute_eigenvectors_from_covariance(seller_data , True , {'total_budget': 2.0, 'delta': 1e-5, 'n': 10000, 'd': 28 * 28})
        
        # Filter eigenvalues and corresponding eigenvectors where eigenvalue > 10^-2
        significant_indices = np.where(seller_eigenvalues > eigenvalue_threshold)[0]
        filtered_eigenvalues = seller_eigenvalues[significant_indices]
        filtered_eigenvectors = seller_eigenvectors[:, significant_indices]
        
        seller_eigenvector_list.append(filtered_eigenvectors)
        seller_eigenvalues_list.append(filtered_eigenvalues)

    # Compute Diversity and Relevance for Each Seller
    diversity_vals, relevance_vals = compute_diversity_and_relevance_for_sellers(
        buyer_images, seller_eigenvector_list, seller_eigenvalues_list)

    # Print seller names and their diversity and relevance values
    for i, txt in enumerate(seller_names):
        print("Seller name:", txt)
        print("Relevance:", relevance_vals[i])
        print("Diversity:", diversity_vals[i])

    # Plot Diversity vs. Relevance
    plot_diversity_vs_relevance(seller_names, diversity_vals, relevance_vals)
    print("Done ")

if __name__ == "__main__":
    main()
