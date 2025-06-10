import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import os
import scipy.stats
import pywt
import scipy.signal as sgn
from scipy.integrate import simpson
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import pandas as pd
import json
#from wavelets.wave_python.waveletFunctions import *
import itertools
from scipy.fftpack import fft
from collections import defaultdict,Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib import cm, ticker
import scipy.io as sio
from IPython.display import display
import datetime as dt
from sklearn.ensemble import GradientBoostingClassifier
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import EngFormatter
import time
import matplotlib.gridspec as gridspec
import math
from tabulate import tabulate
from sklearn.metrics import mean_squared_error
from scipy.signal import firwin, lfilter
from tqdm import tqdm  # For progress tracking
import traceback
from scipy.signal import welch



# Functions to load data
def get_id_cell(sat_id, sat_cell, num_cells=63):    # num_cell = 63 (original)
    id_tx_sat = (sat_id * num_cells) + sat_cell
    return id_tx_sat

def load_data(path, suffix):
    file_ids = os.path.join(path, f"ra_sat_{suffix}.npy")
    file_cells = os.path.join(path, f"ra_cell_{suffix}.npy")

    return np.load(file_ids), np.load(file_cells)

def load_samples(path, suffix):
    samples = os.path.join(path, f"samples_{suffix}.npy")
    fc = os.path.join(path, f"center_frequency_{suffix}.npy")

    return np.load(samples), np.load(fc)


def load_data_all(path, suffixes):
    ids = []
    cells = []
    for suffix in suffixes:
        id_, cell = load_data(path, suffix)
        ids.append(id_)
        cells.append(cell)
    ids_array = np.concatenate(ids)
    cells_array = np.concatenate(cells)
    id_cells_array = get_id_cell(ids_array, cells_array)
    return ids_array, cells_array, id_cells_array

#Functios which is able to manipulate vectors  

def sum_string_vectors(vector1, vector2):

    """
    Sum two vectors of strings element-wise.

    Parameters:
        vector1 (list): First vector of strings.
        vector2 (list): Second vector of strings.

    Returns:
        list: Resulting vector with element-wise string concatenation.
    """
    # Check if the vectors have the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")

    # Concatenate corresponding strings element-wise
    result_vector = [str1 + str2 for str1, str2 in zip(vector1, vector2)]
    
    return result_vector

def add_string_to_vector(initial_string, vector, final_string):
    """
    Add a string to each element of a vector of strings.

    Parameters:
        
        initial_string (str): String to append to each element.
        vector (list): Vector of strings.
        final_string (str): String to append to each element.

    Returns:
        list: Resulting vector with each element modified. result = initial_string  + vector[i] + final_string
    """
    # Use list comprehension to add the additional string to each element
    result_vector = [initial_string + elem + final_string for elem in vector]
    
    return result_vector

# Iterate over each file path if you want to see the information about the dataset
def information_data(file_paths):
    for file_path in file_paths:
        try:
            # Load the .npy file using np.load()
            data = np.load(file_path)

            # Display properties of the loaded data
            print(f"File: {file_path}")
            print("Type of data:", type(data))
            print("Shape of data:", data.shape)
            print("Data type of elements:", data.dtype)
            print("Number of dimensions (axes):", data.ndim)
            print("Total number of elements:", data.size)
            # Example: Calculate and print numerical statistics
            print("Numerical Statistics:")
            print("Min value:", np.min(data))
            print("Max value:", np.max(data))
            print()  # Add a blank line for separation

        except FileNotFoundError:
            print(f"Error: File not found at path: {file_path}")
            print()  # Add a blank line for separation
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            print()  # Add a blank line for separation

def analyze_data_3d(data, sample, plot_name = "Plot"):
    """
    Analyze and visualize data with shape (10000, 11000, 2).

    Parameters:
        data (ndarray): 3D NumPy array with shape (10000, 11000, 2).

    Returns:
        None (displays statistics and plots).
    """
    # Extract dimensions from the data shape
    num_samples = data.shape[0]
    num_features = data.shape[1]
    num_attributes = data.shape[2]

    # Compute mean and standard deviation across samples and features
    mean_values = np.mean(data, axis=(0, 1))
    std_values = np.std(data, axis=(0, 1))

    # Print statistics
    print(f"Number of signals: {num_samples}")
    print(f"Number of samples: {num_features}")
    print(f"IQ Signals: {num_attributes}")
    print(f"Mean values across samples and features: {mean_values}")
    print(f"Standard deviation across samples and features: {std_values}")

    # Flatten data for histogram plotting (combine attributes)
    #flattened_data = data.reshape((num_samples * num_features, num_attributes))

    figsize = 5
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*figsize, 1*figsize))
    ax = axs[0]
    x = np.arange(data.shape[1], dtype=np.float64)
    print(x.shape)
    x /= 25000000
    x *= 1e3
    ax.plot(x, data[sample,:,0], label='Real')
    ax.plot(x, data[sample,:,1], label='Imaginary')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper left')
    ax = axs[1]
    #ax.plot(data[sample,:,0], data[sample,:,1])
    ax.scatter(data[sample,:,0], data[sample,:,1], color='blue', marker='x', s=0.01)
    ax.set_xlabel('Amplitude (Real)')
    ax.set_ylabel('Amplitude (Imaginary)')



 
    plt.savefig(plot_name + '_' + str(sample) + '.png')
    print('File_name: ' + plot_name + '_' + str(sample) + '.png')

def analyze_data_1d(data, size = None ,plot_name = "Plot"):

    if size is None:
        plt.plot(data)
    else:
        plt.plot(data[:size])
    
    num_samples = data.shape[0]
     # Agregar etiquetas y título
    plt.xlabel('Índice')  # Etiqueta del eje x
    plt.ylabel('Valor')    # Etiqueta del eje y
    #plt.title(file_paths[file_number][27:-8])  # Título del gráfico
    plt.title('Atribute') # Título del gráfico
    plt.savefig(plot_name)
    print('data size:{}'.format(data.size))
    print(data)
    print(plot_name)
    return data

def find_not_common_elements(listA, listB):
    # Convert lists to sets
    setA = set(listA)
    setB = set(listB)

    # Find elements exclusive to each set
    not_common_in_A = setA - setB  # Elements in setA but not in setB
    not_common_in_B = setB - setA  # Elements in setB but not in setA
    common_elements = np.intersect1d(listA, listB)
    

    return list(not_common_in_A), list(not_common_in_B), list(common_elements)

def show_unique_and_repeated_elements(lst):
    # Use Counter to count occurrences of each element
    element_counts = Counter(lst)

    # Extract unique elements in ascending order
    unique_elements = sorted(element_counts.keys())

    # Calculate the size of the result vector (number of unique elements)
    vector_size = len(unique_elements)

    # Display unique elements once in ascending order
    print(f"Unique elements in the list - Size: {vector_size}:")
    print(unique_elements)

    # Get repeated elements (count > 1) and sort them
    repeated_elements = {element: count for element, count in element_counts.items() if count > 1}
    
    print("\nRepeated elements in the list:")
    if repeated_elements:
        for element in sorted(repeated_elements.keys()):
            print(f"{element} (Repeated {repeated_elements[element]} times)")
    else:
        print("No repeated elements found.")

    return unique_elements

def numbers_above_threshold(data, threshold):
    # Count occurrences of each number
    counts = Counter(data)
    
    # Filter numbers with frequencies above the threshold
    numbers_above = [(num, freq) for num, freq in counts.items() if freq > threshold]
    
    # Sort numbers by frequency in ascending order
    numbers_above.sort()
    numbers_above = np.array(numbers_above)

    print("Numbers with frequency greater than", threshold, ":", numbers_above[:,0])
    return numbers_above

'''
def find_peaks(signal):
  """
  Finds the peaks (local maxima) in a 1D signal.

  Args:
      signal: A 1D NumPy array representing the signal.

  Returns:
      A list of indices corresponding to the peaks in the signal.
  """

  # Handle potential empty input
  if len(signal) == 0:
    return []

  # Calculate first and second derivatives
  dy = np.diff(signal)
  d2y = np.diff(dy)

  # Find potential peak indices where the second derivative is negative
  # (assuming peaks are convex up)
  peak_indices = np.where(d2y < 0)[0] + 1  # Add 1 to account for the diff

  # Refine peak indices by checking if they are true local maxima
  refined_peaks = []
  for idx in peak_indices:
    if idx > 0 and idx < len(signal) - 1:
      # Check if the current point is greater than its neighbors
      if signal[idx] > signal[idx - 1] and signal[idx] > signal[idx + 1]:
        refined_peaks.append(idx)

  return refined_peaks
'''


def find_peaks_with_frequency(signal, frequency, N_peaks=3, distance=None, prominence=None):
    """
    Finds the N highest peaks in a signal and returns their indices, values, and corresponding frequencies.

    Parameters:
    - signal: List or array of signal values (e.g., amplitude or magnitude over time/frequency).
    - frequency: List or array representing the frequency values corresponding to each point in the signal.
    - N_peaks: Integer, number of peaks to return (default is 3). Peaks are sorted by value in descending order.
    - distance: Optional integer, minimum number of points between detected peaks (default is None).
    - prominence: Optional float, minimum prominence required for a peak to be detected (default is None).

    Returns:
    - sorted_peak_indices: Array of indices of the top N detected peaks in the signal.
    - sorted_peak_values: Array of signal values at those peak indices, sorted from highest to lowest.
    - sorted_peak_frequencies: Array of corresponding frequencies for the detected peaks.
    """
    # Convert the input signal to a NumPy array for efficient numerical operations
    # This ensures compatibility with NumPy functions even if the input is a list
    signal_array = np.asarray(signal)

    # Convert the frequency input to a NumPy array
    # This aligns frequency values with signal values for later mapping
    frequency_array = np.asarray(frequency)

    # Use scipy.signal.find_peaks to detect peaks in the signal
    # - distance: Ensures peaks are separated by at least 'distance' points (optional filtering)
    # - prominence: Filters peaks based on their vertical significance (optional filtering)
    # The function returns peak indices and a dictionary of properties (we ignore the latter with '_')
    peak_indices, _ = sgn.find_peaks(signal_array, distance=distance, prominence=prominence)

    # Check if any peaks were found
    # If peak_indices is empty (size == 0), no peaks meet the criteria
    if peak_indices.size == 0:
        # Print a warning message to inform the user
        print("No peaks found in the signal.")
        # Return empty lists for indices, values, and frequencies as a fallback
        return [], [], []

    # Extract the signal values at the detected peak indices
    # This gives us the height/amplitude of each peak
    peak_values = signal_array[peak_indices]

    # Sort the peaks by their values in descending order
    # np.argsort returns indices that would sort the array; [::-1] reverses for descending order
    sorted_indices = np.argsort(peak_values)[::-1]

    # Reorder the peak indices based on the sorted values
    # This ensures we work with peaks from highest to lowest value
    sorted_peak_indices = peak_indices[sorted_indices]

    # Reorder the peak values to match the sorted indices
    # These are now sorted from highest to lowest
    sorted_peak_values = peak_values[sorted_indices]

    # Map the sorted peak indices to their corresponding frequencies
    # This gives us the frequencies of the peaks in the same order
    sorted_peak_frequencies = frequency_array[sorted_peak_indices]

    # Determine how many peaks to return
    # Take the minimum of N_peaks (user-specified) and the number of detected peaks
    # This prevents indexing errors if fewer peaks are found than requested
    n = min(N_peaks, len(sorted_peak_indices))

    # Return the top N peaks: indices, values, and frequencies
    # Slicing with [:n] ensures we only return up to N_peaks elements
    return sorted_peak_indices[:n], sorted_peak_values[:n], sorted_peak_frequencies[:n]

# Funtions to manipulate data with tensor flow

def explore_tfrecord(tfrecord_file):

    # Open the TFRecord file for reading
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Iterate over the dataset to parse and inspect examples
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Extract features from the example
        feature_dict = example.features.feature
        
        # Print feature keys and types
        
        #[print('id is {}'.format(k)), k=np.array(feature_dict['id'].int64_list.value)[0]]
        id = np.array(feature_dict['id'].int64_list.value)[0]
        id_cell = np.array(feature_dict['id_cell'].int64_list.value)[0]
        cell = np.array(feature_dict['cell'].int64_list.value)[0]
        
        print('id = {} , id_cell = {}, cell = {}'.format(id, id_cell,cell))

        '''
        print("Example Features:")
        for feature_key in feature_dict:
            feature = feature_dict[feature_key]
            feature_type = feature.WhichOneof('kind')
            print(f"- {feature_key}: {feature_type}")
            
            if feature_type == 'bytes_list':
                print(f"  - Values: {feature.bytes_list.value}")
            elif feature_type == 'float_list':
                print(f"  - Values: {feature.float_list.value}")
            elif feature_type == 'int64_list':
                print(f"  - Values: {feature.int64_list.value}")
            # Add handling for other feature types as needed
        '''
        print("-" * 5)

# Graphic functions

def plot_hist(data, bins = 1, title = 'Histogram', col_labels = True, procent= False):
    plt.figure(figsize=(10, 6))  # Tamaño de la figura

    total_count = len(data)

    counts, bins, patches  = plt.hist(data, bins=bins, edgecolor='black')  # Crear el histograma

    if col_labels:
        # Añadir etiquetas de los rangos de los bins
        for bin_start, bin_end in zip(bins[:-1], bins[1:]):
            bin_center = (bin_start + bin_end) / 2
            label = f"{bin_start:.0f} - {bin_end:.0f}"  #f"{ x - y: bin_start - bin_end:.0f}"
            plt.text(bin_center, 0, label, 
                    ha='center', va='top', fontsize=5, rotation=90, color='black')
        
        for count, bin_start, bin_end in zip(counts, bins[:-1], bins[1:]):
            bin_center = (bin_start + bin_end) / 2
            percentage = (count / total_count) * 100

            if procent:
                plt.text(bin_center, count, f'{percentage:.3f}%', 
                    ha='center', va='bottom', fontsize=5, rotation=90)   
            else:
                plt.text(bin_center, count, str(int(count)), 
                    ha='center', va='bottom', fontsize=5, rotation=90)
                
        

    else:
        pass

    plt.xticks([])
    plt.title(title)
    #plt.xlabel("Valor")
    plt.ylabel("Frecuency")
    plt.grid(True)
    plt.show()

def plot_signal_plus_average(ax, time, signal, average_over = 5):
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(average_over))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.set_title('Signal + Time Average', fontsize=16)
    ax.legend(loc='upper right')

def plot_fft_with_bandwidths(freqs: np.ndarray, fft_magnitude: np.ndarray, bandwidths: list, freqs_peaks: np.ndarray, values_peaks: np.ndarray):
    plt.figure(figsize=(15, 4))
    plt.plot(freqs, fft_magnitude, label='FFT Magnitude', color='red')
    
    for low, high in bandwidths:
        plt.fill_between(freqs, fft_magnitude, where=(freqs >= low) & (freqs <= high), color='green', alpha=0.5)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FT with -3 dB Bandwidths')
    plt.legend()
    plt.grid()
    plt.semilogx() 
    plt.semilogy() 

    plt.scatter(freqs_peaks, values_peaks, color='blue', label="Picos detectados", zorder=5)

    plt.show()

def plot_ranges_and_points_log_scale(A, B, figsize=(12, 8), min_log_value=1000):
    """
    Grafica los límites inferiores y superiores de cada vector del arreglo A
    con líneas horizontales una debajo de otra en una escala logarítmica,
    y también marca los valores específicos del vector B.
    
    Parámetros:
    A (list): Lista de tuplas donde cada tupla contiene (límite_inferior, límite_superior)
    B (list/array): Lista o array de valores específicos a marcar en cada rango
    figsize (tuple): Tamaño de la figura (ancho, alto)
    min_log_value (float): Valor mínimo para reemplazar el 0 en la escala logarítmica
    
    Retorna:
    fig, ax: La figura y los ejes de matplotlib
    """
    # Verificar que A y B tienen la misma longitud
    if len(A) != len(B):
        raise ValueError("Los arreglos A y B deben tener la misma longitud")
    
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=figsize)
    
    # Para cada vector en A y valor en B, graficamos sus límites y punto específico
    for i, ((lower, upper), b_value) in enumerate(zip(A, B)):
        # Para el límite inferior, si es 0, usamos 0 visualmente pero min_log_value para el cálculo
        plot_lower = min_log_value if lower == 0 else lower
        
        # Dibujar las líneas horizontales para los rangos
        ax.hlines(i, plot_lower, upper, colors='blue', linewidth=2, alpha=0.7)
        
        # Añadir marcadores en los extremos de los rangos
        ax.plot([plot_lower], [i], 'o', color='red', markersize=6)
        ax.plot([upper], [i], 'o', color='green', markersize=6)
        
        # Añadir el punto específico del vector B
        ax.plot([b_value], [i], 'o', color='purple', markersize=6)
        
        # Añadir texto para indicar cuando el valor es realmente 0
        if lower == 0:
            ax.text(plot_lower*1.2, i, '0', ha='left', va='center', fontsize=9)
    
    # Configurar el eje y
    ax.set_yticks(range(len(A)))
    ax.set_yticklabels([f'Vector {i+1}' for i in range(len(A))])
    
    # Configurar el eje x como escala logarítmica
    ax.set_xscale('log')
    
    # Añadir etiquetas
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Banda')
    ax.set_title('Límites inferiores y superiores con valores específicos en escala logarítmica')
    
    # Añadir leyenda
    ax.plot([], [], 'o', color='red', label='Límite inferior')
    ax.plot([], [], 'o', color='green', label='Límite superior')
    ax.plot([], [], '-', color='blue', label='Rango')
    ax.plot([], [], 'D', color='purple', label='Valor específico (B)')
    ax.legend()
    
    # Mostrar la cuadrícula logarítmica
    ax.grid(True, which="both", ls="-", alpha=1)
    
    # Ajustar el rango del eje x para mostrar bien la escala logarítmica
    max_val = max(max([upper for _, upper in A]), max(B))
    min_val = min([lower if lower > 0 else min_log_value for lower, _ in A])
    ax.set_xlim(min_val/10, max_val*1.5)
    
    # Ajustar los márgenes
    plt.tight_layout()
   
    return fig, ax


def tabla_bandas_transpuesta(A, B):
    data = []
    for i, (tupla, b_valor) in enumerate(zip(A, B), start=1):
        low_limit = tupla[0]
        up_limit = tupla[1]
        bw = up_limit - low_limit
        data.append([f"Band {i}", low_limit, up_limit, bw, b_valor])
    headers = ["Band", "Low Limit", "Up Limit", "BW", "F. Peak"]
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".2f"))
#Funtions for fourier transform

def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave


def get_fft_values(vector, f_s, pos=True, n_fft=None):
    """
    Computes the FFT of a given signal with adjustable resolution and returns frequencies, 
    FFT values, power spectrum, and magnitude.

    Parameters:
    - vector: Input signal (NumPy array)
    - f_s: Sampling frequency in Hz
    - pos: If True, returns only positive frequencies (default: True)
    - n_fft: Number of FFT points for resolution control (default: None, uses 2 * signal length)

    Returns:
    - fft_freqs: Frequencies of the FFT
    - fft_values: Complex FFT values
    - power_spectrum: Corrected power spectrum
    - magnitude: Magnitude of the FFT
    """
    # Get the original length of the input signal
    N = len(vector)

    # Determine the number of FFT points to use
    # If n_fft is not specified, default to twice the signal length for improved resolution
    if n_fft is None:
        n_fft = 2 * N
    # Ensure n_fft is at least the signal length to avoid losing data
    elif n_fft < N:
        n_fft = N

    # Compute the FFT with the specified number of points
    # If n_fft > N, zero-padding is applied automatically
    fft_values = np.fft.fft(vector, n=n_fft)

    # Generate the frequency axis based on n_fft and sampling frequency
    # Resolution is f_s / n_fft, finer with larger n_fft
    fft_freqs = np.fft.fftfreq(n_fft, d=1/f_s)

    # Compute the magnitude spectrum from the complex FFT values
    magnitude = np.abs(fft_values)

    # Compute the power spectrum, normalized by the original signal length
    # Power is proportional to magnitude squared
    power_spectrum = (magnitude ** 2) / N

    # If pos is True, filter to keep only positive frequencies (single-sided spectrum)
    if pos:
        # Create a boolean mask for non-negative frequencies (includes DC at 0 Hz)
        positive_freqs = fft_freqs >= 0

        # Filter all arrays to include only positive frequencies
        fft_values = fft_values[positive_freqs]
        power_spectrum = power_spectrum[positive_freqs]
        magnitude = magnitude[positive_freqs]
        fft_freqs = fft_freqs[positive_freqs]

        # Energy correction for single-sided spectrum
        # Multiply by 2 to account for energy from negative frequencies (except DC and Nyquist)
        power_spectrum *= 2

        # Correct the DC component (no mirror image to double)
        power_spectrum[0] /= 2

        # Correct the Nyquist component if n_fft is even
        # Nyquist frequency (f_s/2) exists only for even-length FFT
        if n_fft % 2 == 0:
            power_spectrum[-1] /= 2  # Last element is Nyquist

    # Return the frequency axis, FFT values, power spectrum, and magnitude
    return fft_freqs, fft_values, power_spectrum, magnitude


#Funtions to extract characteristics

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def power_spectral_density(signal, fs, nperseg=8192):
    """
    Calculate the power spectral density (PSD) and the power of the stochastic process.

    Parameters:
    signal (array-like): Input signal.
    fs (float): Sampling frequency of the signal.
    nperseg (int): Length of each segment for the Welch method.

    Returns:
    tuple: Contains power of the stochastic process (float), list of frequencies,
           list of PSD values, and list of normalized PSD values.
    """
    if not isinstance(signal, (list, np.ndarray)):
        raise ValueError("The signal must be an array-like object.")
    
    if not isinstance(fs, (int, float)):
        raise ValueError("The sampling frequency must be a number.")
    
    if not isinstance(nperseg, int):
        raise ValueError("The nperseg parameter must be an integer.")

    # Calculate PSD using Welch's method
    frequencies, psd = sgn.welch(signal, fs, nperseg=min(nperseg, len(signal)))
    
    # Normalize PSD
    normalized_psd = psd / np.sum(psd)
    
    # Calculate power of the stochastic process
    psp = simpson(psd, x=frequencies)

    # Calculate normalized power of the stochastic process
    n_psp = simpson(normalized_psd, x=frequencies)
    
    return psp, n_psp, np.array(frequencies), np.array(psd), np.array(normalized_psd)

def estimate_iq_imbalance(Y_I, Y_Q):
    """
    Estimate the phase mismatch parameter phi_hat_Q given imbalanced IQ signals.

    Parameters:
    Y_I (ndarray): Imbalanced in-phase component
    Y_Q (ndarray): Imbalanced quadrature component

    Returns:
    float: Estimated phase mismatch parameter phi_hat_Q in degrees
    """
    # Step 1: Calculate Expectations
    E_YI2 = np.mean(Y_I**2)  # Expectation of Y_I squared
    E_YQ2 = np.mean(Y_Q**2)  # Expectation of Y_Q squared
    E_YIYQ = np.mean(Y_I * Y_Q)  # Expectation of the product of Y_I and Y_Q

    # Step 2: Estimate Phase Mismatch Parameter
    phi_hat_Q = np.arcsin(E_YIYQ / np.sqrt(E_YI2 * E_YQ2))  # Estimated phase mismatch for quadrature path
    
    return np.rad2deg(phi_hat_Q)


def get_features(list_values, fs, include_features=['entropy', 'crossings', 'statistics', 'psp']):
    """
    Extract features from the given list of values.
    
    Parameters:
    list_values (list or np.array): Input signal values.
    fs (float): Sampling frequency of the input signal.
    include_features (list of str): List of features to include. Options are 'entropy', 'crossings', 'statistics', and 'psp'.
    
    Returns:
    list: Extracted features based on the specified options.

    Example:
    
    Get only PSP and entropy
    features_psp_entropy = get_features(list_values, fs, include_features=['psp', 'entropy'])
    
    Get only PSP
    features_psp = get_features(list_values, fs, include_features=['psp'])

    """
    try:
        features = []

        if 'entropy' in include_features:
            # Calculate entropy
            entropy = calculate_entropy(list_values)
            features.append(entropy)

        if 'crossings' in include_features:
            # Calculate zero crossings
            crossings = calculate_crossings(list_values)
            features.extend(crossings)

        if 'statistics' in include_features:
            # Calculate basic statistics
            statistics = calculate_statistics(list_values)
            features.extend(statistics)

        if 'psp' in include_features:
            # Calculate power spectral density
            psp, _, _, _,_ = power_spectral_density(list_values, fs)
            features.append(psp)

        return features

    except Exception as e:
        print(f"An error occurred while calculating features: {e}")
        return None

def get_train_test(df, y_col, x_cols, ratio):
    """ 
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test


# functions to extract characteristics Version 1 


def get_fft_values_t(vector, f_s, pos=True, window_type=None):

    """
    Computes the FFT of a signal and returns frequencies, FFT values, power spectrum, and magnitude.
    Handles windowing and corrects DC/Nyquist component scaling.

    Parameters:
    - vector: Input signal (NumPy array).
    - f_s: Sampling frequency (Hz).
    - pos: If True, returns only positive frequencies (default: True).
    - window_type: Windowing type ('hann', 'hamming', or None).


    Returns:
    - fft_freqs: Frequency bins (Hz).
    - fft_values: Complex FFT coefficients.
    - power_spectrum: Normalized power spectrum.
    - magnitude: Scaled FFT magnitude.
    """
    N = len(vector)  # Signal length
    
    # --- Apply windowing ---
    if window_type is not None:
        window_type = window_type.lower()  # Case-insensitive input
        
        # Select window function
        if window_type == 'hann':
            window = np.hanning(N)
        elif window_type == 'hamming':
            window = np.hamming(N)
        else:
            raise ValueError("Unsupported window. Use 'hann', 'hamming', or None.")
        
        # Apply window and compute correction factors
        vector_windowed = vector * window
        S1 = np.sum(window)    # Window sum (amplitude correction)
        S2 = np.sum(window**2) # Window energy (power correction)
    else:
        # No windowing (equivalent to rectangular window)
        vector_windowed = vector
        S1 = N  # Sum of ones (rectangular window)
        S2 = N  # Energy of ones (rectangular window)

    # --- Compute FFT ---
    fft_values = np.fft.fft(vector_windowed)  # Complex FFT coefficients
    fft_freqs = np.fft.fftfreq(N, d=1/f_s)    # Frequency bins

    # --- Normalize spectra ---
    magnitude = np.abs(fft_values) / S1   # Amplitude-corrected magnitude
    power_spectrum = (np.abs(fft_values)**2) / S2  # Energy-corrected power

    # --- Positive frequency handling ---
    if pos:
        # Create mask for non-negative frequencies
        positive_mask = fft_freqs >= 0
        
        # Filter all outputs
        fft_freqs = fft_freqs[positive_mask]
        fft_values = fft_values[positive_mask]
        magnitude = magnitude[positive_mask]
        power_spectrum = power_spectrum[positive_mask]

        # --- Energy correction for single-sided spectrum ---
        # Multiply all components by 2 (except DC and Nyquist)
        power_spectrum *= 2
        
        # Correct DC component (0 Hz)
        power_spectrum[0] /= 2  # DC has no mirror image
        
        # Correct Nyquist component (if present)
        if N % 2 == 0:  # Nyquist exists only for even-length signals
            power_spectrum[-1] /= 2  # Last component is Nyquist (f_s/2)

    return fft_freqs, fft_values, power_spectrum, magnitude


def detect_peaks_t(fft_freqs: np.ndarray, fft_magnitude: np.ndarray, alpha: float = 0.1, beta: int = 5) -> np.ndarray:
    """
    Detects spectral peaks using an adaptive threshold method.
    
    Parameters:
    - fft_freqs: Array of frequencies from FFT.
    - fft_magnitude: Magnitude spectrum.
    - alpha: Threshold factor for peak detection, relative to the maximum magnitude.
    - beta: Minimum peak separation in indices to avoid detecting closely spaced noise peaks.
    
    Returns:
    - peak_freqs: Array of detected peak frequencies.
    """
    height_threshold = alpha * np.max(fft_magnitude)  # Adaptive threshold based on max peak value
    peak_indices, _ = sgn.find_peaks(fft_magnitude, height=height_threshold, distance=beta)
    return fft_freqs[peak_indices]

'''
def compute_bandwidth_t(fft_freqs: np.ndarray, fft_magnitude: np.ndarray, peak_freqs: np.ndarray) -> List[Tuple[float, float]]:
    """
    Computes the -3 dB bandwidth for each detected peak using interpolation.
    
    Parameters:
    - fft_freqs: Array of frequencies from FFT.
    - fft_magnitude: Magnitude spectrum.
    - peak_freqs: Array of detected peak frequencies.
    
    Returns:
    - List of tuples (low_freq, high_freq) representing the -3 dB bandwidths of each peak.
    """
    bandwidths = []
    for peak in peak_freqs:
        peak_index = np.argmin(np.abs(fft_freqs - peak))  # Find closest index to peak frequency
        half_power = fft_magnitude[peak_index] / 2  # Compute half-power (-3 dB) threshold
        low_idx = np.where(fft_magnitude[:peak_index] <= half_power)[0][-1]  # Find lower bound
        high_idx = np.where(fft_magnitude[peak_index:] <= half_power)[0][0] + peak_index  # Find upper bound
        bandwidths.append((fft_freqs[low_idx], fft_freqs[high_idx]))
    return bandwidths
'''

def compute_bandwidth_t(fft_freqs: np.ndarray, fft_magnitude: np.ndarray, peak_freqs: np.ndarray, umbral = 0.707) -> list:
    """
    Calcula el ancho de banda a -3 dB para cada pico detectado usando interpolación lineal.

    Parámetros:
    - fft_freqs: Array de frecuencias del espectro (en orden ascendente).
    - fft_magnitude: Magnitud del espectro correspondiente a cada frecuencia.
    - peak_freqs: Array de frecuencias donde se detectaron los picos.

    Retorna:
    - Lista de tuplas, donde cada tupla tiene las frecuencias inferior y superior del ancho de banda.
    """
    bandwidths = []

    for peak in peak_freqs:
        # Encuentra el índice más cercano a la frecuencia del pico
        peak_index = np.argmin(np.abs(fft_freqs - peak))
        half_power = fft_magnitude[peak_index] * umbral  # Umbral de -3 dB

        # --- Búsqueda del límite inferior (izquierda del pico) ---
        low_freq = None
        for i in range(peak_index, -1, -1):  # Desde el pico hacia el inicio
            if fft_magnitude[i] <= half_power:
                if i < peak_index:
                    # Interpola entre i y i+1
                    x0, x1 = fft_freqs[i], fft_freqs[i + 1]
                    y0, y1 = fft_magnitude[i], fft_magnitude[i + 1]
                    low_freq = x0 + (half_power - y0) * (x1 - x0) / (y1 - y0)
                else:
                    low_freq = fft_freqs[peak_index]
                break
        if low_freq is None:
            low_freq = fft_freqs[0]  # Usa el inicio si no hay cruce

        # --- Búsqueda del límite superior (derecha del pico) ---
        high_freq = None
        for j in range(peak_index, len(fft_freqs)):  # Desde el pico hacia el final
            if fft_magnitude[j] <= half_power:
                if j > peak_index:
                    # Interpola entre j-1 y j
                    x0, x1 = fft_freqs[j - 1], fft_freqs[j]
                    y0, y1 = fft_magnitude[j - 1], fft_magnitude[j]
                    high_freq = x0 + (half_power - y0) * (x1 - x0) / (y1 - y0)
                else:
                    high_freq = fft_freqs[peak_index]
                break
        if high_freq is None:
            high_freq = fft_freqs[-1]  # Usa el final si no hay cruce

        bandwidths.append((low_freq, high_freq))

    return bandwidths

def fir_hamming_filters(signal_data: np.ndarray, fs: float, bandwidths: List[Tuple[float, float]], 
                              ripple_db: float = 0.5, attenuation_db: float = 60, 
                              transition_width_percent: float = 0.1) -> List[np.ndarray]:
    """
    Applies FIR bandpass filters with a Hamming window to an IQ signal, optimized for RF fingerprinting.

    Parameters:
    - signal_data: Input signal (numpy array, can be complex for IQ data).
    - fs: Sampling frequency in Hz (must be 25e6 Hz as per requirements).
    - bandwidths: List of tuples containing low and high cutoff frequencies in Hz, e.g., [(f_low1, f_high1), ...].
    - ripple_db: Maximum ripple in the passband in dB (default is 0.5).
    - attenuation_db: Minimum attenuation in the stopband in dB (default is 60).
    - transition_width_percent: Percentage of the passband width used for the transition width (default is 10%).

    Returns:
    - A list of filtered signals, each corresponding to a pair of cutoff frequencies in bandwidths.
    """
    # Check if the sampling frequency matches the required 25e6 Hz
    if fs != 25e6:
        raise ValueError("The sampling frequency must be 25e6 Hz as per the requirements.")

    # Initialize an empty list to store the filtered signals for each bandwidth
    filtered_signals = []
    
    # Calculate the Nyquist frequency (half of the sampling frequency)
    nyq = fs / 2

    # Convert the ripple and attenuation from dB to linear scale for filter design
    delta_p = 10 ** (-ripple_db / 20) - 1  # Passband ripple in linear scale (e.g., ~0.057 for 0.5 dB)
    delta_s = 10 ** (-attenuation_db / 20)  # Stopband attenuation in linear scale (e.g., 0.001 for 60 dB)

    # Loop through each pair of low and high cutoff frequencies in the bandwidths list
    for f_low, f_high in bandwidths:
        # Validate that the cutoff frequencies are within acceptable bounds
        if not (0 < f_low < f_high < nyq):
            raise ValueError(f"The cutoff frequencies ({f_low}, {f_high}) must satisfy 0 < f_low < f_high < {nyq} Hz.")

        # Compute the passband width (difference between high and low cutoff frequencies)
        passband_width = f_high - f_low
        if passband_width <= 0:
            raise ValueError(f"The passband width ({passband_width}) must be positive.")

        # Calculate the transition width as a percentage of the passband width
        transition_width = transition_width_percent * passband_width
        
        # Normalize the transition width relative to the sampling frequency
        delta_f = transition_width / fs

        # Estimate the filter order (N) for a Hamming window based on the normalized transition width
        N = int(np.ceil(3.3 / delta_f))
        
        # Ensure the filter order is odd to maintain linear phase characteristics
        if N % 2 == 0:
            N += 1

        # Verify that the calculated filter order is valid (positive)
        if N <= 0:
            raise ValueError(f"The calculated order ({N}) is invalid. Adjust transition_width_percent.")

        # Design the FIR bandpass filter coefficients using a Hamming window
        # Frequencies are normalized by the Nyquist frequency for the firwin function
        coefficients = firwin(N, [f_low / nyq, f_high / nyq], pass_zero=False, window='hamming')

        # Apply the filter to the input signal using convolution
        filtered_signal = lfilter(coefficients, 1.0, signal_data)
        
        # Add the filtered signal to the list
        filtered_signals.append(filtered_signal)

    # Return the list of all filtered signals
    return filtered_signals

def optimized_fir_hamming_filters(signal_data: np.ndarray, fs: float, 
                                 bandwidths: List[Tuple[float, float]], 
                                 ripple_db: float = 0.5, 
                                 attenuation_db: float = 60, 
                                 transition_width_percent: float = 0.1) -> List[np.ndarray]:
    """
    Applies FIR bandpass filters with a Hamming window to an IQ signal, optimized for RF fingerprinting.
    This version ensures linear convolution and is optimized for speed.
    
    Parameters:
    - signal_data: Input signal (numpy array, can be complex for IQ data).
    - fs: Sampling frequency in Hz (must be 25e6 Hz as per requirements).
    - bandwidths: List of tuples containing low and high cutoff frequencies in Hz.
    - ripple_db: Maximum ripple in the passband in dB (default is 0.5, unused in firwin).
    - attenuation_db: Minimum attenuation in the stopband in dB (default is 60, unused in firwin).
    - transition_width_percent: Percentage of the passband width used for the transition width (default is 10%).
    
    Returns:
    - A list of filtered signals, each corresponding to a pair of cutoff frequencies in bandwidths.
    """
    # Validate sampling frequency
    if fs != 25e6:
        raise ValueError("The sampling frequency must be 25e6 Hz as per the requirements.")
    
    nyq = fs / 2  # Nyquist frequency
    
    # Step 1: Compute filter order N for each bandwidth to find the maximum
    Ns = []
    for f_low, f_high in bandwidths:
        if not (0 < f_low < f_high < nyq):
            raise ValueError(f"The cutoff frequencies ({f_low}, {f_high}) must satisfy 0 < f_low < f_high < {nyq} Hz.")
        passband_width = f_high - f_low
        if passband_width <= 0:
            raise ValueError(f"The passband width ({passband_width}) must be positive.")
        transition_width = transition_width_percent * passband_width
        delta_f = transition_width / fs
        N = int(np.ceil(3.3 / delta_f))  # Filter order based on Hamming window
        if N % 2 == 0:
            N += 1  # Ensure odd order for linear phase
        if N <= 0:
            raise ValueError(f"The calculated order ({N}) is invalid. Adjust transition_width_percent.")
        Ns.append(N)
    
    max_N = max(Ns)  # Maximum filter order
    
    # Step 2: Set FFT size to support linear convolution for all filters
    min_nfft = len(signal_data) + max_N - 1
    nfft = 2 ** int(np.ceil(np.log2(min_nfft)))  # Next power of 2 for efficiency
    
    # Step 3: Compute signal FFT once
    signal_fft = np.fft.fft(signal_data, nfft)
    
    # Step 4: Filter each bandwidth
    filtered_signals = []
    filter_cache = {}
    
    for (f_low, f_high), N in zip(bandwidths, Ns):
        cache_key = (f_low, f_high, N)
        if cache_key in filter_cache:
            freq_response = filter_cache[cache_key]
        else:
            # Design FIR bandpass filter
            coefficients = firwin(N, [f_low / nyq, f_high / nyq], pass_zero=False, window='hamming')
            # Compute filter FFT, zero-padding to nfft
            freq_response = np.fft.fft(coefficients, nfft)
            filter_cache[cache_key] = freq_response
        
        # Apply filter in frequency domain
        filtered_fft = signal_fft * freq_response
        # Inverse FFT and truncate to input signal length
        filtered_signal = np.fft.ifft(filtered_fft)[:len(signal_data)].real
        filtered_signals.append(filtered_signal)
    
    return filtered_signals


def compute_psd_t(signal_data, fs, bandwidths: List[Tuple[float, float]], nperseg=16384) -> List[np.ndarray]:
    """
    Computes the Power Spectral Density (PSD) for each detected frequency band using Welch's method.
    
    Parameters:
    - signal_data: signal_data object containing IQ signal and sampling frequency.
    - bandwidths: List of (low_freq, high_freq) tuples defining frequency bands.
    
    Returns:
    - List of PSD arrays corresponding to each frequency band.
    """
    f, Pxx = sgn.welch(signal_data, fs, nperseg=nperseg)  # Compute Welch PSD
    psd_bands = [Pxx[(f >= low) & (f <= high)] for low, high in bandwidths]  # Filter PSD for selected bands
    return psd_bands


def compute_psd_v(senales_filtradas: List[np.ndarray], fs: float, nperseg: int = 16384) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Computes the Power Spectral Density (PSD) of each filtered signal using the Welch method.

    Parameters:
    - senales_filtradas: List of filtered signals (each one is a NumPy array).
    - fs: Sampling frequency in Hz.
    - nperseg: Number of points per segment for Welch's PSD calculation (default is 16384).

    Returns:
    - psd_freqs: List of frequency arrays corresponding to the PSD calculation for each signal.
    - psd_bands: List of PSD arrays, where each entry corresponds to a filtered signal.
    """
    psd_bands = []  # List to store the computed PSD values
    psd_freqs = []  # List to store the corresponding frequency values

    for senal in senales_filtradas:
        # Compute the PSD using Welch’s method
        freqs, psd = sgn.welch(senal, fs, nperseg=min(nperseg, len(senal)))

        # Append the PSD values to the list
        psd_bands.append(psd)

        # Append the corresponding frequency values
        psd_freqs.append(freqs)
    
    return psd_freqs, psd_bands

def extract_features_t(psd_bands: List[np.ndarray], bandwidths: List[Tuple[float, float]], freq_peaks: List[float], mag_peaks: List[float]) -> List[dict]:
    """
    Extracts key spectral features from each frequency band, including band limits and peaks.

    Parameters:
    - psd_bands: List of PSD arrays corresponding to detected frequency bands.
    - bandwidths: List of tuples (low_frequency, high_frequency) for each band.
    - freq_peaks: List of FFT peak frequencies (in Hz), one per band.
    - mag_peaks: List of FFT peak magnitude , one per band.
    
    Returns:
    - List of dictionaries containing extracted spectral features for each band.
    """
    features = []
    
    # Check that psd_bands and bandwidths have the same length
    if len(psd_bands) != len(bandwidths):
        raise ValueError("psd_bands and bandwidths must have the same length")
    if freq_peaks is not None and len(freq_peaks) != len(psd_bands):
        raise ValueError("freq_peaks must match the length of psd_bands if provided")
    if mag_peaks is not None and len(mag_peaks) != len(psd_bands):
        raise ValueError("mag_peaks must match the length of psd_bands if provided")

    # Loop through each PSD band, its frequency limits, and optional peak
    for i, (psd, bw) in enumerate(zip(psd_bands, bandwidths)):
        n = len(psd)

        if n < 2:
            # Insufficient points for frequency-based features
            features.append({
                "power": 0.0,
                "entropy": np.nan,
                "spectral_centroid": np.nan,
                "psd_mean": np.mean(psd) if n > 0 else np.nan,
                "psd_variance": np.var(psd) if n > 0 else np.nan,
                "l_band": bw[0],
                "h_band": bw[1],
                "frequency_peak": freq_peaks[i],
                "Magnitude_peak": mag_peaks[i]
            })
            continue
        
        ######## TOTAL POWER BY INTEGTATING PSD OVER THE FREQUENCY BAND VIA TRAPEZOIDAL INTEGRATION ########

        # Calculate frequency resolution for the band
        delta_f = (bw[1] - bw[0]) / (n - 1)
        # Calculate power
        power = np.trapz(psd, dx=delta_f)
        
        ######## SPECTRAL ENTROPY WITH NORMALIZAED PSD ########
        if np.sum(psd) > 0:
            psd_norm = psd / np.sum(psd)
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
        else:
            entropy = np.nan
        
        ######## SPECTRAL CENTROID IN FREQUENCY UNITS ########
        if np.sum(psd) > 0:
            centroid_index = np.sum(psd * np.arange(n)) / np.sum(psd)
            f_centroid = bw[0] + centroid_index * delta_f
        else:
            f_centroid = np.nan
        
        ######## MEAN AND VARIANCE OF PSD VALUES ########
        psd_mean = np.mean(psd)
        psd_variance = np.var(psd)
              
        # Store features in a dictionary
        features.append({
            "power": power,                     # Total power of the stochastic process in the PSD band
            "entropy": entropy,                 # Normalized spectral entropy
            "spectral_centroid": f_centroid,    # Spectral centroid (Hz)
            "psd_mean": psd_mean,               # Mean of PSD values
            "psd_variance": psd_variance,       # Variance of PSD values
            "l_band": bw[0],                    # Lower frequency limit (Hz)
            "h_band": bw[1],                    # Upper frequency limit (Hz)
            "frequency_peak": freq_peaks[i],        # FFT peak frequency (in Hz)
            "Magnitude_peak": mag_peaks[i]          # FFT peak magnitude
        })
    
    return features

def extract_features_v(psd_bands, psd_freqs, freq_peaks, mag_peaks, bandwidths) -> List[dict]:
    """
    Extracts key spectral features from each frequency band, including band limits and peaks.

    Parameters:
    - psd_bands: List of PSD arrays corresponding to detected frequency bands.
    - psd_freqs: List of frequency arrays corresponding to each PSD band.
    - freq_peaks: List of FFT peak frequencies (in Hz), one per band.
    - mag_peaks: List of FFT peak magnitudes, one per band.
    - bandwidths: List of tuples (low_frequency, high_frequency) for each band.
    
    Returns:
    - List of dictionaries containing extracted spectral features for each band.
    """
    features = []  # Initialize an empty list to store feature dictionaries for each band
    
    # Input validation to ensure consistent lengths across provided lists
    if len(psd_bands) != len(psd_freqs):
        raise ValueError("psd_bands and psd_freqs must have the same length")
    if freq_peaks is not None and len(freq_peaks) != len(psd_bands):
        raise ValueError("freq_peaks must match the length of psd_bands if provided")
    if mag_peaks is not None and len(mag_peaks) != len(psd_bands):
        raise ValueError("mag_peaks must match the length of psd_bands if provided")

    # Loop through each PSD band, its corresponding frequencies, and bandwidth limits
    for i, (psd, freq, bw) in enumerate(zip(psd_bands, psd_freqs, bandwidths)):
        n = len(psd)  # Number of data points in the current PSD band

        # Handle cases with insufficient data points (less than 2)
        if n < 2:
            # Insufficient points for frequency-based features like power, entropy, and centroid
            features.append({
                "power": 0.0,                    # Total power set to zero due to insufficient data
                "entropy": np.nan,               # Entropy cannot be computed, set to NaN
                "spectral_centroid": np.nan,     # Spectral centroid cannot be computed, set to NaN
                "psd_mean": np.mean(psd) if n > 0 else np.nan,  # Mean of PSD, or NaN if no data
                "psd_variance": np.var(psd) if n > 0 else np.nan,  # Variance of PSD, or NaN if no data
                "l_band": bw[0],                 # Lower frequency limit of the band (Hz)
                "h_band": bw[1],                 # Upper frequency limit of the band (Hz)
                "frequency_peak": freq_peaks[i], # FFT peak frequency for this band (Hz)
                "Magnitude_peak": mag_peaks[i]   # FFT peak magnitude for this band
            })
            continue  # Skip to the next band
        
        # Calculate total power by integrating PSD over the frequency band using trapezoidal integration
        power = np.trapz(psd, freq)  # Integrates PSD values over the frequency range
        
        # Compute spectral centroid and entropy if power is positive
        if power > 0:
            # Spectral centroid: frequency-weighted average of PSD, normalized by total power
            f_centroid = np.trapz(freq * psd, freq) / power

            # Normalize PSD for entropy calculation
            psd_norm = psd / power
            # Compute the integrand for entropy: -p * log2(p), with small epsilon to avoid log(0)
            integrand = -psd_norm * np.log2(psd_norm + 1e-12)
            # Integrate to calculate spectral entropy
            entropy = np.trapz(integrand, freq)
        else:
            # If power is zero or negative, set centroid and entropy to NaN
            f_centroid = np.nan
            entropy = np.nan
        
        # Calculate mean and variance of the PSD values
        psd_mean = np.mean(psd)      # Arithmetic mean of PSD values
        psd_variance = np.var(psd)   # Variance of PSD values to measure spread
        
        # Store all computed features in a dictionary for this band
        features.append({
            "power": power,                  # Total power of the stochastic process in the PSD band
            "entropy": entropy,              # Normalized spectral entropy, a measure of spectral complexity
            "spectral_centroid": f_centroid, # Spectral centroid in Hz, the "center of mass" of the spectrum
            "psd_mean": psd_mean,            # Mean of PSD values in the band
            "psd_variance": psd_variance,    # Variance of PSD values, indicating variability
            "l_band": bw[0],                 # Lower frequency limit of the band (Hz)
            "h_band": bw[1],                 # Upper frequency limit of the band (Hz)
            "frequency_peak": freq_peaks[i], # FFT peak frequency in this band (Hz)
            "Magnitude_peak": mag_peaks[i]   # FFT peak magnitude in this band
        })
    
    return features  # Return the list of feature dictionaries for all bands


def standardize_features(features_list):
    """
    Estandariza las características de una lista de diccionarios.
    
    Args:
        features_list (list): Lista de diccionarios con características numéricas.
    
    Returns:
        list: Lista de diccionarios con las características estandarizadas.
    """
    if not features_list:
        return []
    
    # Obtener las claves (características) del primer diccionario
    keys = features_list[0].keys()
    
    # Recolectar todos los valores para cada característica
    values_dict = {key: [] for key in keys}
    for feature_dict in features_list:
        for key in keys:
            values_dict[key].append(feature_dict[key])
    
    # Calcular media y desviación estándar para cada característica
    means = {key: np.mean(values_dict[key]) for key in keys}
    stds = {key: np.std(values_dict[key]) for key in keys}
    
    # Estandarizar los valores
    standardized_list = []
    for feature_dict in features_list:
        standardized_dict = {}
        for key in keys:
            if stds[key] != 0:  # Evitar división por cero
                standardized_value = (feature_dict[key] - means[key]) / stds[key]
            else:
                standardized_value = 0  # Si std es 0, asignamos 0 (todos los valores son iguales)
            standardized_dict[key] = standardized_value
        standardized_list.append(standardized_dict)
    
    return standardized_list

def create_feature_matrix(magnitude_dict, phase_dict, sat_label, transmitter_label, feature_counter, normalize = False):
    """
    Creates a concatenated feature matrix for a neural network from magnitude and phase dictionaries.
    
    Args:
        magnitude_dict (list): List of dictionaries with magnitude features.
        phase_dict (list): List of dictionaries with phase features.
        sat_label (int): Integer label for the satellite.
        transmitter_label (int): Integer label for the transmitter.
        feature_counter (int): Consecutive integer to name the feature matrix (e.g., features_001).
        normalize: if the data is normalized or not
    Returns:
        tuple: (feature_matrix, metadata) where feature_matrix is a NumPy array and metadata is a dictionary.
    """
    
    # Extract number of bands (rows) from dictionaries
    num_magnitude_bands = len(magnitude_dict)
    num_phase_bands = len(phase_dict)
    
    # Extract feature names from the first dictionary (assuming all have the same structure)
    feature_names = list(magnitude_dict[0].keys())
    num_features = len(feature_names)
    
    # Initialize matrices for magnitude and phase features
    magnitude_matrix = np.zeros((num_magnitude_bands, num_features))
    phase_matrix = np.zeros((num_phase_bands, num_features))
    
    # Populate magnitude matrix
    for i, band in enumerate(magnitude_dict):
        magnitude_matrix[i] = [band[feat] for feat in feature_names]
    
    # Populate phase matrix
    for i, band in enumerate(phase_dict):
        phase_matrix[i] = [band[feat] for feat in feature_names]
    
    # Concatenate magnitude and phase matrices (magnitude rows first, then phase rows)
    feature_matrix = np.vstack((magnitude_matrix, phase_matrix))
    
    # Name the feature matrix with a consecutive integer (e.g., features_001)
    #feature_matrix_name = f"features_{feature_counter:03d}"
    feature_matrix_name = f"features_{feature_counter}"

    # Create metadata dictionary
    metadata = {
        "Satellite_Label": sat_label,
        "Transmitter_Label": transmitter_label,
        "Feature_Matrix_Name": feature_matrix_name,
        "Normalized": str(normalize),
        "Colum_Feature_Names": feature_names,
        "Rows": [f"Magnitude_Band_{i+1}" for i in range(num_magnitude_bands)] + 
                [f"Phase_Band_{i+1}" for i in range(num_phase_bands)],
        "Description": f"Concatenated matrix with {num_magnitude_bands} magnitude bands in the first rows and {num_phase_bands} phase bands in the last rows."
    }
    
    return feature_matrix, metadata

def iq_features_processor(i_signal, q_signal, fs=25e6, number_peaks=10, normalize =False ):
    """
        This function takes in-phase (I) and quadrature (Q) components of a signal, computes features 
        related to magnitude and phase, and returns both raw and standardized feature sets. It is 
        designed to analyze signal characteristics in the frequency domain, potentially for use 
        in machine learning models such as neural networks.
        Args:
            i_signal (array-like): In-phase component of the signal, typically a time-domain array.
            q_signal (array-like): Quadrature component of the signal, typically a time-domain array.
            fs (float, optional): Sampling frequency in Hz, defaults to 25 MHz (25e6 Hz).
            number_peaks (int, optional): Number of peaks to detect or process in feature extraction, defaults to 10.
            normalize (bool, optional): If True, standardize (normalize) the extracted features. Defaults to False.

        Returns:
        tuple: A 2-element tuple containing:
            - features_mag (list of dict): List of dictionaries with magnitude-related features 
                                          (e.g., power, entropy, spectral_centroid) for each band or segment.
            - features_phase (list of dict): List of dictionaries with phase-related features 
                                            (e.g., instant frequency stats) for each band or segment.

        Notes:
            - The function assumes I and Q signals are synchronized and of the same length.
            - Features are derived from spectral or statistical analysis of the combined IQ signal.
            - The sampling frequency (fs) is used to compute frequency-domain features.
            - The number_peaks parameter influences peak-related features (e.g., frequency_peak, magnitude_peak).
            - The normalize parameter controls whether the features are standardized after extraction.

    """    

    dt = 1/fs #period

    # Calculate the magnitude and phase of the IQ signal
    mag = np.sqrt(q_signal**2 + i_signal**2)
    phase = np.arctan2(q_signal, i_signal)

    # ---------- COMPUTE THE ELECTIONS OF MAGNITUDE - INSTANTANEUS FREQUENCY BANDS ---------- 

    aggregated_vector_mag = np.ones(5500)*-10000
    aggregated_vector_phase = np.ones(5500)*-10000

    # Unwrap phase to avoid discontinuities
    phase_unwrapped = np.unwrap(phase)
    # Apply Savitzky-Golay filter to smooth phase and calculate the instantaneous frequency
    phase_diff = savgol_filter(phase_unwrapped, window_length=5, polyorder=2, deriv=1, delta=dt)

    # Get the FFT values for the magnitude of the signal
    freq_values_mag, fft_values_mag, power_spectrum_mag, magnitude_mag = get_fft_values(mag, fs, pos=True)
    aggregated_vector_mag = np.amax(np.vstack([aggregated_vector_mag, magnitude_mag]), axis=0)

    # Get the FFT values for the instantaneus frequency
    freq_values_phase, fft_values_phase, power_spectrum_phase, magnitude_phase  = get_fft_values(phase_diff, fs, pos=True)
    aggregated_vector_phase = np.amax(np.vstack([aggregated_vector_phase, magnitude_phase]), axis=0)

    # Find the peaks in the aggregated vectors

    indices_mag, values_mag, freqs_mag = find_peaks_with_frequency(aggregated_vector_mag, freq_values_mag, N_peaks=number_peaks)
    indices_phase, values_phase, freqs_phase = find_peaks_with_frequency(aggregated_vector_phase, freq_values_phase, N_peaks=number_peaks)

    #print(f"{number_peaks:03d} Peaks calculated ")

    # Bandwidth for magnitude 
    bandwidths_mag = compute_bandwidth_t(freq_values_mag, aggregated_vector_mag, freqs_mag)
    #plot_fft_with_bandwidths(freq_values_mag, aggregated_vector_mag, bandwidths_mag, freqs_mag, values_mag)

    # Bandwidth for instant frequency 
    bandwidths_phase = compute_bandwidth_t(freq_values_phase, aggregated_vector_phase, freqs_phase)
    #plot_fft_with_bandwidths(freq_values_phase, aggregated_vector_phase, bandwidths_phase, freqs_phase, values_phase)

    #print("Magnitud and Instantaneus frequency Bands Calculated")

    # ---------- end compute the elections of magnitude - instantaneus frequency bands ---------- 


    # ---------- COMPUTE POWER SPECTRAL DENSITY PER BAND ---------- 

    nperseg=11000

    # Compute de Magnitude PSD for each band calculated
    psd_bands_mag = compute_psd_t(mag, fs, bandwidths_mag, nperseg=nperseg)

    # Compute de instantaneous frequency PSD for each band calculated
    psd_bands_phase = compute_psd_t(phase_diff, fs, bandwidths_phase, nperseg=nperseg)
    
    #print("Power Spectrum Density per Band Calculated")

    # ---------- end calculations of power spectral density per band ---------- 

    # ---------- COMPUTE MAGNITUDE - INSTANTANEUS FREQUENCY FEATURES  ---------- 

    # Extract spectral features
    features_mag = extract_features_t(psd_bands_mag, bandwidths_mag, freqs_mag, values_mag)
    features_phase = extract_features_t(psd_bands_phase, bandwidths_phase, freqs_phase, values_phase)
    #print("Features calculated")
    
    if normalize:
        # ---------- COMPUTE STANDARDIZE FEATURES  ---------- 

        features_mag = standardize_features(features_mag)
        features_phase = standardize_features(features_phase)
        #print("Standardize Features calculated")

        # ---------- end compute standardize features  ---------- 

    # ---------- end compute magnitude - isntantaneus frequency features  ---------- 

    return features_mag, features_phase

def iq_features_processor_v(i_signal, q_signal, fs=25e6, number_peaks=10, normalize =False, bw_umbral = 0.707 ):
    """
        This function takes in-phase (I) and quadrature (Q) components of a signal, computes features 
        related to magnitude and phase, and returns both raw and standardized feature sets. It is 
        designed to analyze signal characteristics in the frequency domain, potentially for use 
        in machine learning models such as neural networks.
        Args:
            i_signal (array-like): In-phase component of the signal, typically a time-domain array.
            q_signal (array-like): Quadrature component of the signal, typically a time-domain array.
            fs (float, optional): Sampling frequency in Hz, defaults to 25 MHz (25e6 Hz).
            number_peaks (int, optional): Number of peaks to detect or process in feature extraction, defaults to 10.
            normalize (bool, optional): If True, standardize (normalize) the extracted features. Defaults to False.

        Returns:
        tuple: A 2-element tuple containing:
            - features_mag (list of dict): List of dictionaries with magnitude-related features 
                                          (e.g., power, entropy, spectral_centroid) for each band or segment.
            - features_phase (list of dict): List of dictionaries with phase-related features 
                                            (e.g., instant frequency stats) for each band or segment.

        Notes:
            - The function assumes I and Q signals are synchronized and of the same length.
            - Features are derived from spectral or statistical analysis of the combined IQ signal.
            - The sampling frequency (fs) is used to compute frequency-domain features.
            - The number_peaks parameter influences peak-related features (e.g., frequency_peak, magnitude_peak).
            - The normalize parameter controls whether the features are standardized after extraction.

    """    

    dt = 1/fs #period

    # Calculate the magnitude and phase of the IQ signal
    mag = np.sqrt(q_signal**2 + i_signal**2)
    phase = np.arctan2(q_signal, i_signal)

    # ---------- COMPUTE THE ELECTIONS OF MAGNITUDE - INSTANTANEUS FREQUENCY BANDS ---------- 

    N = len(mag)
    n_fft = 1

    # Unwrap phase to avoid discontinuities
    phase_unwrapped = np.unwrap(phase)
    # Apply Savitzky-Golay filter to smooth phase and calculate the instantaneous frequency
    phase_diff = savgol_filter(phase_unwrapped, window_length=5, polyorder=2, deriv=1, delta=dt)

    # Get the FFT values for the magnitude of the signal
    freq_values_mag, _, _, magnitude_mag = get_fft_values(mag, fs, pos=True, n_fft=n_fft)
    aggregated_vector_mag = magnitude_mag

    # Get the FFT values for the instantaneus frequency
    freq_values_phase, _, _, magnitude_phase  = get_fft_values(phase_diff, fs, pos=True, n_fft=n_fft)
    aggregated_vector_phase = magnitude_phase

    # Find the peaks in the aggregated vectors

    _, values_mag, freqs_mag = find_peaks_with_frequency(aggregated_vector_mag, freq_values_mag, N_peaks=number_peaks)
    _, values_phase, freqs_phase = find_peaks_with_frequency(aggregated_vector_phase, freq_values_phase, N_peaks=number_peaks)

    #print(f"{number_peaks:03d} Peaks calculated ")
    umbral = bw_umbral
    # Bandwidth for magnitude 
    bandwidths_mag = compute_bandwidth_t(freq_values_mag, aggregated_vector_mag, freqs_mag, umbral)
    #plot_fft_with_bandwidths(freq_values_mag, aggregated_vector_mag, bandwidths_mag, freqs_mag, values_mag)

    # Bandwidth for instant frequency 
    bandwidths_phase = compute_bandwidth_t(freq_values_phase, aggregated_vector_phase, freqs_phase, umbral)
    #plot_fft_with_bandwidths(freq_values_phase, aggregated_vector_phase, bandwidths_phase, freqs_phase, values_phase)

    #print("Magnitud and Instantaneus frequency Bands Calculated")

    # ---------- end compute the elections of magnitude - instantaneus frequency bands ---------- 


    # ---------- COMPUTE POWER SPECTRAL DENSITY PER BAND ---------- 

    nperseg=16384

    # Compute FIR Filters per each bandwidth Magnitude
    senales_filtradas_mag = fir_hamming_filters(mag, fs, bandwidths_mag)

    # Compute FIR Filters per each bandwidth Instant Frequency
    senales_filtradas_phase = fir_hamming_filters(phase_diff, fs, bandwidths_phase) 


    # Compute de Magnitude PSD for each band calculated
    psd_freqs_mag, psd_bands_mag = compute_psd_v(senales_filtradas_mag, fs, nperseg=nperseg)


    # Compute de instantaneous frequency PSD for each band calculated
    psd_freqs_phase, psd_bands_phase = compute_psd_v(senales_filtradas_phase, fs, nperseg=nperseg)
    
    #print("Power Spectrum Density per Band Calculated")

    # ---------- end calculations of power spectral density per band ---------- 

    # ---------- COMPUTE MAGNITUDE - INSTANTANEUS FREQUENCY FEATURES  ---------- 

        # Extract spectral features
    features_mag = extract_features_v(psd_bands_mag, psd_freqs_mag, freqs_mag, values_mag, bandwidths_mag)
    features_phase = extract_features_v(psd_bands_phase, psd_freqs_phase, freqs_phase, values_phase, bandwidths_phase)
    #print("Features calculated")
    
    if normalize:
        # ---------- COMPUTE STANDARDIZE FEATURES  ---------- 

        features_mag = standardize_features(features_mag)
        features_phase = standardize_features(features_phase)
        #print("Standardize Features calculated")

        # ---------- end compute standardize features  ---------- 

    # ---------- end compute magnitude - isntantaneus frequency features  ---------- 

    return features_mag, features_phase

def iq_features_processor_v1(i_signal, q_signal, fs=25e6, number_peaks=10, normalize =False, bw_umbral = 0.707 ):
    """
        This function takes in-phase (I) and quadrature (Q) components of a signal, computes features 
        related to magnitude and phase, and returns both raw and standardized feature sets. It is 
        designed to analyze signal characteristics in the frequency domain, potentially for use 
        in machine learning models such as neural networks.
        Args:
            i_signal (array-like): In-phase component of the signal, typically a time-domain array.
            q_signal (array-like): Quadrature component of the signal, typically a time-domain array.
            fs (float, optional): Sampling frequency in Hz, defaults to 25 MHz (25e6 Hz).
            number_peaks (int, optional): Number of peaks to detect or process in feature extraction, defaults to 10.
            normalize (bool, optional): If True, standardize (normalize) the extracted features. Defaults to False.

        Returns:
        tuple: A 2-element tuple containing:
            - features_mag (list of dict): List of dictionaries with magnitude-related features 
                                          (e.g., power, entropy, spectral_centroid) for each band or segment.
            - features_phase (list of dict): List of dictionaries with phase-related features 
                                            (e.g., instant frequency stats) for each band or segment.

        Notes:
            - The function assumes I and Q signals are synchronized and of the same length.
            - Features are derived from spectral or statistical analysis of the combined IQ signal.
            - The sampling frequency (fs) is used to compute frequency-domain features.
            - The number_peaks parameter influences peak-related features (e.g., frequency_peak, magnitude_peak).
            - The normalize parameter controls whether the features are standardized after extraction.

    """    

    dt = 1/fs #period

    # Calculate the magnitude and phase of the IQ signal
    mag = np.sqrt(q_signal**2 + i_signal**2)
    phase = np.arctan2(q_signal, i_signal)

    # ---------- COMPUTE THE ELECTIONS OF MAGNITUDE - INSTANTANEUS FREQUENCY BANDS ---------- 

    N = len(mag)
    n_fft = 1

    # Unwrap phase to avoid discontinuities
    phase_unwrapped = np.unwrap(phase)
    # Apply Savitzky-Golay filter to smooth phase and calculate the instantaneous frequency
    phase_diff = savgol_filter(phase_unwrapped, window_length=5, polyorder=2, deriv=1, delta=dt)

    # Get the FFT values for the magnitude of the signal
    freq_values_mag, _, _, magnitude_mag = get_fft_values(mag, fs, pos=True, n_fft=n_fft)
    aggregated_vector_mag = magnitude_mag

    # Get the FFT values for the instantaneus frequency
    freq_values_phase, _, _, magnitude_phase  = get_fft_values(phase_diff, fs, pos=True, n_fft=n_fft)
    aggregated_vector_phase = magnitude_phase

    # Find the peaks in the aggregated vectors

    _, values_mag, freqs_mag = find_peaks_with_frequency(aggregated_vector_mag, freq_values_mag, N_peaks=number_peaks)
    _, values_phase, freqs_phase = find_peaks_with_frequency(aggregated_vector_phase, freq_values_phase, N_peaks=number_peaks)

    #print(f"{number_peaks:03d} Peaks calculated ")
    umbral = bw_umbral
    # Bandwidth for magnitude 
    bandwidths_mag = compute_bandwidth_t(freq_values_mag, aggregated_vector_mag, freqs_mag, umbral)
    #plot_fft_with_bandwidths(freq_values_mag, aggregated_vector_mag, bandwidths_mag, freqs_mag, values_mag)

    # Bandwidth for instant frequency 
    bandwidths_phase = compute_bandwidth_t(freq_values_phase, aggregated_vector_phase, freqs_phase, umbral)
    #plot_fft_with_bandwidths(freq_values_phase, aggregated_vector_phase, bandwidths_phase, freqs_phase, values_phase)

    #print("Magnitud and Instantaneus frequency Bands Calculated")

    # ---------- end compute the elections of magnitude - instantaneus frequency bands ---------- 


    # ---------- COMPUTE POWER SPECTRAL DENSITY PER BAND ---------- 

    nperseg=16384

    # Compute FIR Filters per each bandwidth Magnitude
    senales_filtradas_mag = optimized_fir_hamming_filters(mag, fs, bandwidths_mag)

    # Compute FIR Filters per each bandwidth Instant Frequency
    senales_filtradas_phase = optimized_fir_hamming_filters(phase_diff, fs, bandwidths_phase) 


    # Compute de Magnitude PSD for each band calculated
    psd_freqs_mag, psd_bands_mag = compute_psd_v(senales_filtradas_mag, fs, nperseg=nperseg)


    # Compute de instantaneous frequency PSD for each band calculated
    psd_freqs_phase, psd_bands_phase = compute_psd_v(senales_filtradas_phase, fs, nperseg=nperseg)
    
    #print("Power Spectrum Density per Band Calculated")

    # ---------- end calculations of power spectral density per band ---------- 

    # ---------- COMPUTE MAGNITUDE - INSTANTANEUS FREQUENCY FEATURES  ---------- 

        # Extract spectral features
    features_mag = extract_features_v(psd_bands_mag, psd_freqs_mag, freqs_mag, values_mag, bandwidths_mag)
    features_phase = extract_features_v(psd_bands_phase, psd_freqs_phase, freqs_phase, values_phase, bandwidths_phase)
    #print("Features calculated")
    
    if normalize:
        # ---------- COMPUTE STANDARDIZE FEATURES  ---------- 

        features_mag = standardize_features(features_mag)
        features_phase = standardize_features(features_phase)
        #print("Standardize Features calculated")

        # ---------- end compute standardize features  ---------- 

    # ---------- end compute magnitude - isntantaneus frequency features  ---------- 

    return features_mag, features_phase

def get_signals(df, sat_id=None, cell_id=None):
    """
    Returns the signals associated with a satellite (sat_id), a cell (cell_id), or both.
    The filtering is done on a DataFrame that already contains 'ra_sat' and 'ra_cell' as columns,
    preserving the original row order.
    
    Args:
        df (pandas.DataFrame): DataFrame with a MultiIndex (ra_sat, ra_cell) and a 'samples' column.
        sat_id (int, optional): Satellite ID to filter by.
        cell_id (int, optional): Cell ID to filter by.
    
    Returns:
        Depending on the provided filters:
          - Both provided: a list of samples for the specific (sat_id, cell_id) combination.
          - Only sat_id: a list of (cell_id, samples) pairs.
          - Only cell_id: a list of (sat_id, samples) pairs.
          - None if no criteria or no matching data.
    """
    # If no filter is provided, return None.
    if sat_id is None and cell_id is None:
        return None

    # Use a temporary DataFrame that has 'ra_sat' and 'ra_cell' as columns.
    # If these columns already exist, no need to reset the index.
    if 'ra_sat' in df.columns and 'ra_cell' in df.columns:
        tmp = df
    else:
        tmp = df.reset_index()

    # If both sat_id and cell_id are provided, filter by both.
    if sat_id is not None and cell_id is not None:
        filtered = tmp[(tmp['ra_sat'] == sat_id) & (tmp['ra_cell'] == cell_id)]
        return filtered['samples'].tolist() if not filtered.empty else None

    # If only sat_id is provided, filter by sat_id and return (cell_id, samples) pairs.
    if sat_id is not None:
        filtered = tmp[tmp['ra_sat'] == sat_id]
        return list(zip(filtered['ra_cell'], filtered['samples'])) if not filtered.empty else None

    # If only cell_id is provided, filter by cell_id and return (sat_id, samples) pairs.
    if cell_id is not None:
        filtered = tmp[tmp['ra_cell'] == cell_id]
        return list(zip(filtered['ra_sat'], filtered['samples'])) if not filtered.empty else None


# ---------------functions to extract characteristics Version 2 --------------------------

# - Variance Fractal Dimension Trajectory (Per τ and T)
#     - Variance (Overall)
# - Phase Noise (Per τ and T) (Jitter)
#     - Variance (Overall)
# - sample variance of the fractional frequency fluctuations (FC and FS) (Per τ and T)
#     - Allan Deviation: how much the oscillators wander in the time scale (Overall)
#     - Flicker Noise ??? 
# - IQ Imbalance (Per τ and T)
#     - IQ Imbalance (Overall)
# - Phasor Noise Deviation (Per τ and T)  (PA Non Linearities)
#     - Variance (Overall)

def get_signals_v2(df: pd.DataFrame, sat_id=None, cell_id=None):
    """
    Returns the signals and center‐frequency values associated with a satellite (sat_id),
    a cell (cell_id), or both.

    Args:
        df (pd.DataFrame): DataFrame indexed or with columns 'ra_sat', 'ra_cell',
                           containing columns 'fcs' and 'samples'.
        sat_id (int, optional): Satellite ID to filter by.
        cell_id (int, optional): Cell ID to filter by.

    Returns:
        - If both sat_id and cell_id: list of (fc, samples) tuples.
        - If only sat_id:           list of (cell_id, fc, samples) tuples.
        - If only cell_id:          list of (sat_id, fc, samples) tuples.
        - None if no criteria given or no matches found.
    """
    # No filter means nothing to return
    if sat_id is None and cell_id is None:
        return None

    # Make sure 'ra_sat', 'ra_cell', 'fcs' are columns for easy filtering
    if {'ra_sat', 'ra_cell', 'fcs'}.issubset(df.columns):
        tmp = df
    else:
        tmp = df.reset_index()

    # Both satellite and cell specified
    if sat_id is not None and cell_id is not None:
        filtered = tmp[(tmp['ra_sat'] == sat_id) & (tmp['ra_cell'] == cell_id)]
        if filtered.empty:
            return None
        # Return just (fc, samples)
        return list(zip(filtered['fcs'], filtered['samples']))

    # Only satellite specified
    if sat_id is not None:
        filtered = tmp[tmp['ra_sat'] == sat_id]
        if filtered.empty:
            return None
        # Return (cell_id, fc, samples)
        return list(zip(filtered['ra_cell'], filtered['fcs'], filtered['samples']))

    # Only cell specified
    if cell_id is not None:
        filtered = tmp[tmp['ra_cell'] == cell_id]
        if filtered.empty:
            return None
        # Return (sat_id, fc, samples)
        return list(zip(filtered['ra_sat'], filtered['fcs'], filtered['samples']))

def create_metadata_from_features_dict(features_dict: dict,
                                       sat_label: int,
                                       transmitter_label: int,
                                       feature_counter: int,
                                       fs: float,
                                       win_length: int,
                                       win_shift: int,
                                       fc: float
                                      ) -> dict:
    """
    Generate metadata for a pre-computed features dictionary without altering it.

    Args:
      features_dict: dict mapping feature_name -> 1D array of values
      sat_label: Integer label for the satellite
      transmitter_label: Integer label for the transmitter
      feature_counter: Integer used to name this feature set (e.g. 2 → "features_2")
      fs: Sampling frequency in Hz
      win_length: Window length in samples
      win_shift: Window shift in samples
      fc: Center frequency in Hz

    Returns:
      metadata: dict containing only JSON-serializable types:
        - Satellite_Label (int)
        - Transmitter_Label (int)
        - Feature_Dictionary_Name (str)
        - Sampling_Frequency (float)
        - Win_Length (int)
        - Win_Shift (int)
        - Center_Frequency (float)
        - Rows (list of str)
        - Row_Lengths (list of int)
    """
    # Cast numeric inputs to built-in types to ensure JSON serializability
    sat_label = int(sat_label)
    transmitter_label = int(transmitter_label)
    fs = float(fs)
    win_length = int(win_length)
    win_shift = int(win_shift)
    fc = float(fc)

    feature_dictionary_name = f"features_{int(feature_counter)}"
    rows = list(map(str, features_dict.keys()))
    # Ensure row lengths are Python ints
    row_lengths = [int(len(features_dict[row])) for row in rows]

    metadata = {
        "Satellite_Label":        sat_label,
        "Transmitter_Label":      transmitter_label,
        "Feature_Dictionary_Name": feature_dictionary_name,
        "Sample_Frequency":     fs,
        "Win_Length":             win_length,
        "Win_Shift":              win_shift,
        "Center_Frequency":       fc,
        "Rows":                   rows,
        "Row_Lengths":            row_lengths
    }
    return metadata

def vfdt(window_mag: np.ndarray,
            window_phase: np.ndarray,
            win_length: int):
    """
    Compute VFDT values for a single window of magnitude and unwrapped phase.

    Parameters:
      window_mag (np.ndarray): magnitude samples for this window, length L
      window_phase (np.ndarray): unwrapped phase samples for this window, length L
      win_length (int): window length L

    Returns:
      vfdt_magnitude (float)
      vfdt_phase     (float)
    """
    # magnitude differences
    diff_mag = window_mag[1:] - window_mag[:-1]
    var_mag  = np.var(diff_mag)
    if var_mag <= 0:
        vfdt_magnitude = 2.0
    else:
        vfdt_magnitude = 2 - (np.log(var_mag) / (2 * np.log(win_length)))

    # phase differences
    diff_phase = window_phase[1:] - window_phase[:-1]
    var_phase  = np.var(diff_phase)
    if var_phase <= 0:
        vfdt_phase = 2.0
    else:
        vfdt_phase = 2 - (np.log(var_phase) / (2 * np.log(win_length)))

    return vfdt_magnitude, vfdt_phase

def phase_noise(window_i: np.ndarray,
                window_q: np.ndarray,
                fs: float,
                remove_amplitude_variation: bool = True,
                polynomial_degree: int = 1) -> float:
    """
    Compute the variance of detrended, unwrapped phase (“phase noise”) from I/Q samples.

    Parameters
    ----------
    window_i : np.ndarray
        In-phase samples (1D).
    window_q : np.ndarray
        Quadrature samples (1D), same length as window_i.
    fs : float
        Sampling frequency in Hz (must be > 0).
    remove_amplitude_variation : bool
        If True, normalize out amplitude modulation before phase extraction.
    polynomial_degree : int
        Degree of polynomial to fit & remove (must be ≥ 0).

    Returns
    -------
    float
        Variance of the detrended, unwrapped phase.
    """
    # ——— Validation ———
    if window_i.ndim != 1 or window_q.ndim != 1:
        raise ValueError("window_i and window_q must be 1-D arrays")
    if window_i.shape != window_q.shape:
        raise ValueError("window_i and window_q must have the same length")
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive")
    
    N = window_i.size
    # Cap polynomial_degree so polyfit doesn’t blow up
    if polynomial_degree < 0:
        raise ValueError("polynomial_degree must be non-negative")
    if polynomial_degree >= N:
        polynomial_degree = N - 1

    # ——— Build complex signal ———
    cplx = window_i + 1j * window_q

    # ——— 1) Remove amplitude variation ———
    if remove_amplitude_variation:
        mags = np.abs(cplx)
        cplx = cplx / (mags + 1e-12)

    # ——— 2) Instantaneous phase & unwrap ———
    phase = np.angle(cplx)
    phase_unwrapped = np.unwrap(phase)

    # ——— 3) Polynomial detrend of phase ———
    t = np.arange(N) / fs
    # fit & subtract
    p_fit = np.polyfit(t, phase_unwrapped, deg=polynomial_degree)
    phase_fit = np.polyval(p_fit, t)
    phase_detrended = phase_unwrapped - phase_fit

    ph_noise = np.var(phase_detrended)

    # ——— 4) Return variance ———
    return ph_noise

def sample_variance_and_allan(y: np.ndarray,
                              fs: float,
                              win_length: int,
                              win_shift: int) -> Tuple[float, np.ndarray, float]:
    """
    Compute time-domain sample variance and Allan deviation of fractional frequency fluctuations.

    Parameters
    ----------
    y : np.ndarray
        1D array of fractional frequency deviation samples.
    fs : float
        Sampling rate in Hz.
    win_length : int
        Window length (samples) for averaging (m).
    win_shift : int
        Shift (samples) between consecutive windows (M).

    Returns
    -------
    sigma2 : float
        Unbiased sample variance of block-averaged deviations.
    y_bar : np.ndarray
        Block-averaged fractional deviations (length N).
    sigma_y : float
        Allan deviation computed as sqrt(0.5 * mean(diff(y_bar)**2)).
    """
    # Input validation
    if win_length <= 0 or win_shift <= 0:
        raise ValueError("win_length and win_shift must be positive integers")
    if win_length > len(y):
        raise ValueError("win_length is larger than the length of y")
    
    # Compute averaging and repetition intervals
    tau = win_length / fs
    T   = win_shift / fs

    # Number of full, non-overlapping windows
    N = 1 + (len(y) - win_length) // win_shift
    if N < 1:
        raise ValueError("Not enough data points for even one window")
    
    # Compute block averages ȳ_k
    y_bar = np.array([
        y[k*win_shift : k*win_shift + win_length].mean()
        for k in range(N)
    ])
    
    # Sample variance (unbiased)
    sigma2 = np.var(y_bar, ddof=1) if N > 1 else 0.0

    # Allan deviation σ_y = sqrt(0.5 * mean((ȳ_{k+1}-ȳ_k)^2))
    if N > 1:
        diffs = np.diff(y_bar)
        sigma2_allan = 0.5 * np.mean(diffs**2)
        sigma_y = np.sqrt(sigma2_allan)
    else:
        sigma_y = 0.0

    return y_bar, sigma2, sigma_y

def IQ_feature_processor_V3(i_signal: np.ndarray,
                            q_signal: np.ndarray,
                            win_length: int,
                            win_shift: int,
                            fs: float,
                            fc:float,
                            pn_band: Tuple[float, float] = (1e3, 1e5)
                           ) -> dict:
    
    # Calculate the magnitude and phase of the IQ signal
    complex_signal = i_signal + 1j * q_signal
    mag = np.abs(complex_signal)
    phase = np.angle(complex_signal)

    # Unwrap phase to remove discontinuities at ±π
    phase_unwrapped = np.unwrap(phase)

    f_inst = np.diff(phase) * fs * ( 1 / (2 * np.pi))  # gamma*
    y = (f_inst - fc) / fc # fractional instantaneous frequency deviation (y)

    M = (len(mag) - win_length) // win_shift + 1
    features = {k: np.zeros(M) for k in [
        "vfdt_mag", 
        "vfdt_ph", 
        "noise_ph",
        "frac_freq_var",
        "win_idx"
    ]}

    idx = 0
    for start in range(0, len(mag) - win_length + 1, win_shift):
        end = start + win_length
        win_i = i_signal[start:end]
        win_q = q_signal[start:end]
        win_mag = mag[start:end]
        win_ph  = phase_unwrapped[start:end]

        # VFDT
        features["vfdt_mag"][idx], features["vfdt_ph"][idx] = vfdt(win_mag, win_ph, win_length)

        # PHASE NOISE
        features["noise_ph"][idx] = phase_noise(win_i, win_q, fs, remove_amplitude_variation = True, polynomial_degree = 1)

        # WINDONW INDEX
        features["win_idx"][idx] = start

        idx += 1
    
    # FRACTIONAL FREQUENCY FLUCTUATIONS
    y_bar, sigma2, sigma_y = sample_variance_and_allan(y, fs, win_length, win_shift)
    features["frac_freq_var"] = y_bar

    M = len(y_bar)
    features["frac_freq_sigma2"]     = np.full(1, sigma2)
    features["frac_freq_allan_dev"]  = np.full(1, sigma_y)    

    return features
