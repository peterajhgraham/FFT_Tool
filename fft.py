import numpy as np
from numba import jit

@jit(nopython=True)
def fft(x):
    """
    Compute the Fast Fourier Transform of an input array using the Cooley-Tukey algorithm.
    This implementation is optimized for large datasets through the usage of NumPy and Numba for JIT compilation.
    
    Parameters:
        x (numpy.ndarray): Input array containing the signal to be transformed.

    Returns:
        numpy.ndarray: The FFT of the input array.
    """
    N = x.shape[0]

    if N <= 1:
        return x

    # Recursively apply FFT to even and odd indexed elements
    even = fft(x[::2])
    odd = fft(x[1::2])

    # Combine results using the FFT formula
    T = np.exp(-2j * np.pi * np.arange(N) / N)[:N // 2] * odd
    return np.concatenate([even + T, even - T])

def load_large_dataset(filepath):
    """
    Load a large dataset from a NumPy binary file.
    
    Parameters:
        filepath (str): The path to the .npy file.

    Returns:
        numpy.ndarray: The loaded dataset.
    """
    return np.load(filepath)

def save_fft_result(result, output_filepath):
    """
    Save the FFT result to a file.
    
    Parameters:
        result (numpy.ndarray): The FFT result.
        output_filepath (str): The path where the result will be saved.
    """
    np.save(output_filepath, result)

if __name__ == "__main__":
    # Load the large dataset
    dataset = load_large_dataset('large_dataset.npy')

    # Compute the FFT
    fft_result = fft(dataset)

    # Save the result to a file
    save_fft_result(fft_result, 'fft_result.npy')
