import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.stats import poisson


# Alternative: using Gaussian filter (faster for large sigma_s)
def discrete_vote_bilateral(image, gray_values, sigma_s=1.5, sigma_r=10):
    """
    Same as above but uses gaussian_filter (separable, much faster for large kernels).
    """
    support_maps = []
    for g in gray_values:
        intensity_likelihood = poisson.pmf(image, g)
        # gaussian_filter automatically uses separable kernels
        support = gaussian_filter(intensity_likelihood, sigma=sigma_s, mode='reflect')
        support_maps.append(support)
    
    support_stack = np.array(support_maps)
    best_indices = np.argmax(support_stack, axis=0)
    result = np.array(gray_values)[best_indices]
    return result

# Example usage
if __name__ == "__main__":
    # Create a test image with two materials (0 and 255) plus Gaussian noise
    np.random.seed(0)
    phantom = np.zeros((128, 128))
    phantom[32:96, 32:96] = 255
    noise = np.random.normal(0, 20, phantom.shape)  # sigma_r ≈ 20
    noisy_image = phantom + noise
    
    gray_vals = [0, 255]
    # Use sigma_r slightly higher than noise to be tolerant
    segmented = discrete_vote_bilateral(noisy_image, gray_vals, sigma_s=2.0, sigma_r=25)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(131); plt.imshow(noisy_image, cmap='gray'); plt.title('Noisy image')
    plt.subplot(132); plt.imshow(segmented, cmap='gray'); plt.title('Segmented')
    plt.subplot(133); plt.imshow(phantom, cmap='gray'); plt.title('Original')
    plt.show()