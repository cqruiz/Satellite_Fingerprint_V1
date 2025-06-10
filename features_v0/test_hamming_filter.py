import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import unittest
from typing import List, Tuple
import time

from scipy.signal import firwin, lfilter

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

# For comparison, we'll include the original function
def original_fir_hamming_filters(signal_data: np.ndarray, fs: float, bandwidths: List[Tuple[float, float]], 
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

class TestOptimizedFIRHammingFilters(unittest.TestCase):
    def setUp(self):
        # Set up test parameters
        self.fs = 25e6  # 25 MHz sampling rate
        self.duration = 0.001  # 1 ms duration
        self.num_samples = int(self.fs * self.duration)
        self.t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        self.transition_width_percent = 0.1  # Add this line to fix the error
        
        # Define test bandwidths
        self.bandwidths = [
            (1e6, 3e6),    # 1-3 MHz
            (5e6, 8e6),    # 5-8 MHz
            (9e6, 10e6)    # 9-10 MHz
        ]
        
        # Generate test signals
        self.generate_test_signals()
    
    def generate_test_signals(self):
        """Generate test signals with components at different frequencies."""
        # Create a multi-frequency test signal
        f1, f2, f3, f4 = 2e6, 7e6, 9.5e6, 12e6  # Frequencies in Hz
        
        # Complex signal with components at different frequencies
        self.test_signal = (
            np.sin(2 * np.pi * f1 * self.t) +  # 2 MHz component (in first band)
            0.5 * np.sin(2 * np.pi * f2 * self.t) +  # 7 MHz component (in second band)
            0.3 * np.sin(2 * np.pi * f3 * self.t) +  # 9.5 MHz component (in third band)
            0.2 * np.sin(2 * np.pi * f4 * self.t)    # 12 MHz component (outside all bands)
        )
        
        # Add some noise
        self.test_signal += 0.1 * np.random.randn(len(self.test_signal))
    
    def test_filter_output_shape(self):
        """Test if the optimized filter produces the expected output shape."""
        filtered_signals = optimized_fir_hamming_filters(self.test_signal, self.fs, self.bandwidths)
        
        # Check if the output is a list
        self.assertIsInstance(filtered_signals, list)
        
        # Check if there's one output for each bandwidth
        self.assertEqual(len(filtered_signals), len(self.bandwidths))
        
        # Check if each output has the correct length
        for signal in filtered_signals:
            self.assertEqual(len(signal), len(self.test_signal))
    
    def test_filter_frequency_response(self):
        """Test if the optimized filter properly isolates frequency components."""
        filtered_signals = optimized_fir_hamming_filters(self.test_signal, self.fs, self.bandwidths)
        
        # Analyze frequency content of each filtered signal
        for i, signal in enumerate(filtered_signals):
            f_low, f_high = self.bandwidths[i]
            
            # Compute power spectrum
            spectrum = np.abs(np.fft.fft(signal)) ** 2
            freqs = np.fft.fftfreq(len(signal), 1/self.fs)
            
            # Check positive frequencies within the band
            in_band_mask = (freqs > f_low) & (freqs < f_high)
            out_band_mask = (freqs > 0) & ~in_band_mask & (freqs < self.fs/2)
            
            if np.sum(in_band_mask) > 0 and np.sum(out_band_mask) > 0:
                # Calculate average power in-band and out-of-band
                avg_power_in_band = np.mean(spectrum[in_band_mask])
                avg_power_out_band = np.mean(spectrum[out_band_mask])
                
                # In-band power should be significantly higher than out-of-band
                self.assertGreater(avg_power_in_band, 10 * avg_power_out_band)
    
    def test_against_original_implementation(self):
        """Test if the optimized filter produces results similar to the original."""
        original_results = original_fir_hamming_filters(self.test_signal, self.fs, self.bandwidths)
        optimized_results = optimized_fir_hamming_filters(self.test_signal, self.fs, self.bandwidths)
        
        # Check if results are close enough
        for i in range(len(self.bandwidths)):
            # Compare original and optimized results
            # We expect small differences due to FFT-based implementation vs direct convolution
            # But the overall characteristics should be similar
            correlation = np.abs(np.corrcoef(np.real(original_results[i]), np.real(optimized_results[i]))[0, 1])
            self.assertGreater(correlation, 0.95)  # Correlation should be high
    
    def test_filter_error_handling(self):
        """Test if the filter correctly handles invalid inputs."""
        # Test with invalid sampling frequency
        with self.assertRaises(ValueError):
            optimized_fir_hamming_filters(self.test_signal, 24e6, self.bandwidths)
        
        # Test with invalid bandwidth (low > high)
        invalid_bandwidths = [(5e6, 3e6)]
        with self.assertRaises(ValueError):
            optimized_fir_hamming_filters(self.test_signal, self.fs, invalid_bandwidths)
        
        # Test with negative frequency
        invalid_bandwidths = [(-1e6, 3e6)]
        with self.assertRaises(ValueError):
            optimized_fir_hamming_filters(self.test_signal, self.fs, invalid_bandwidths)
        
        # Test with frequency beyond Nyquist
        invalid_bandwidths = [(1e6, 15e6)]
        with self.assertRaises(ValueError):
            optimized_fir_hamming_filters(self.test_signal, self.fs, invalid_bandwidths)

def visualize_filter_performance(test_signal, fs, bandwidths, transition_width_percent=0.1):
    """
    Helper function to visualize and compare filter performance between original and optimized implementations.
    """
    # Get filtered signals from both implementations
    original_filtered = original_fir_hamming_filters(test_signal, fs, bandwidths, transition_width_percent=transition_width_percent)
    optimized_filtered = optimized_fir_hamming_filters(test_signal, fs, bandwidths, transition_width_percent=transition_width_percent)
    
    # Create time axis
    t = np.linspace(0, len(test_signal)/fs, len(test_signal), endpoint=False)
    
    # Create frequency axis for FFT plots
    freqs = np.fft.fftfreq(len(test_signal), 1/fs)
    pos_freqs_mask = freqs >= 0
    
    # Plot time domain signals
    plt.figure(figsize=(15, 12))
    
    # Original signal
    plt.subplot(len(bandwidths)+1, 2, 1)
    plt.plot(t[:1000], test_signal[:1000])
    plt.title('Original Signal (Time Domain)')
    plt.xlabel('Time (s)')
    
    # Original spectrum
    plt.subplot(len(bandwidths)+1, 2, 2)
    spectrum = np.abs(np.fft.fft(test_signal))
    plt.plot(freqs[pos_freqs_mask]/1e6, spectrum[pos_freqs_mask])
    plt.title('Original Signal (Frequency Domain)')
    plt.xlabel('Frequency (MHz)')
    plt.xlim(0, fs/2e6)
    
    # Filtered signals
    for i in range(len(bandwidths)):
        # Time domain comparison
        plt.subplot(len(bandwidths)+1, 2, 3+i*2)
        plt.plot(t[:1000], np.real(original_filtered[i][:1000]), 'b-', label='Original')
        plt.plot(t[:1000], np.real(optimized_filtered[i][:1000]), 'r--', label='Optimized')
        plt.title(f'Filtered Signal {i+1}: {bandwidths[i][0]/1e6}-{bandwidths[i][1]/1e6} MHz (Time)')
        plt.xlabel('Time (s)')
        plt.legend()
        
        # Frequency domain comparison
        plt.subplot(len(bandwidths)+1, 2, 4+i*2)
        spectrum_orig = np.abs(np.fft.fft(original_filtered[i]))
        spectrum_opt = np.abs(np.fft.fft(optimized_filtered[i]))
        plt.plot(freqs[pos_freqs_mask]/1e6, spectrum_orig[pos_freqs_mask], 'b-', label='Original')
        plt.plot(freqs[pos_freqs_mask]/1e6, spectrum_opt[pos_freqs_mask], 'r--', label='Optimized')
        plt.axvspan(bandwidths[i][0]/1e6, bandwidths[i][1]/1e6, alpha=0.3, color='green')
        plt.title(f'Filtered Signal {i+1}: {bandwidths[i][0]/1e6}-{bandwidths[i][1]/1e6} MHz (Frequency)')
        plt.xlabel('Frequency (MHz)')
        plt.xlim(0, fs/2e6)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_bode_diagram(fs, bandwidths, transition_width_percent=0.1):
    """
    Create two Bode diagrams:
    1. All bands of each method together (original above, optimized below)
    2. Each band separately with both methods together in each band
    
    Parameters:
    - fs: Sampling frequency in Hz
    - bandwidths: List of tuples containing low and high cutoff frequencies in Hz
    - transition_width_percent: Percentage of the passband width used for transition
    """
    nyq = fs / 2
    
    # Create frequency axis for Bode plot (logarithmic scale)
    # Use enough points for a smooth curve
    freq_points = 10000
    w = np.logspace(3, np.log10(nyq), freq_points)  # from 1 kHz to Nyquist
    
    # Normalize frequencies for freqz
    w_normalized = w / fs
    
    # Store frequency responses for all bands
    original_responses = []
    optimized_responses = []
    
    # Calculate filter responses for each band
    for i, (f_low, f_high) in enumerate(bandwidths):
        # Calculate filter parameters
        passband_width = f_high - f_low
        transition_width = transition_width_percent * passband_width
        delta_f = transition_width / fs
        
        # Estimate filter order
        N = int(np.ceil(3.3 / delta_f))
        if N % 2 == 0:
            N += 1
        
        # Design filter - both methods use the same coefficients
        coefficients = signal.firwin(N, [f_low / nyq, f_high / nyq], pass_zero=False, window='hamming')
        
        # Compute frequency response
        _, h = signal.freqz(coefficients, worN=w_normalized * 2 * np.pi)
        
        # Store responses
        original_responses.append(h)
        optimized_responses.append(h)  # Same coefficients for both methods
    
    # PLOT 1: All bands of each method together
    plt.figure(figsize=(12, 10))
    
    # Plot original method (top)
    plt.subplot(2, 1, 1)
    for i, h in enumerate(original_responses):
        f_low, f_high = bandwidths[i]
        plt.semilogx(w, 20 * np.log10(np.abs(h)), label=f'Band {i+1}: {f_low/1e6:.1f}-{f_high/1e6:.1f} MHz')
        
        # Highlight the passband
        plt.axvspan(f_low, f_high, alpha=0.1, color=f'C{i}')
    
    plt.title('All Bands - Original Method')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(-80, 5)
    
    # Plot optimized method (bottom)
    plt.subplot(2, 1, 2)
    for i, h in enumerate(optimized_responses):
        f_low, f_high = bandwidths[i]
        plt.semilogx(w, 20 * np.log10(np.abs(h)), label=f'Band {i+1}: {f_low/1e6:.1f}-{f_high/1e6:.1f} MHz')
        
        # Highlight the passband
        plt.axvspan(f_low, f_high, alpha=0.1, color=f'C{i}')
    
    plt.title('All Bands - Optimized Method')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(-80, 5)
    
    plt.tight_layout()
    plt.show()
    
    # PLOT 2: Each band separately with both methods together
    plt.figure(figsize=(15, 5 * len(bandwidths)))
    
    for i, (f_low, f_high) in enumerate(bandwidths):
        # Plot magnitude response
        plt.subplot(len(bandwidths), 2, i*2+1)
        
        # Plot original method
        plt.semilogx(w, 20 * np.log10(np.abs(original_responses[i])), 'b-', 
                     label='Original Method')
        
        # Plot optimized method
        plt.semilogx(w, 20 * np.log10(np.abs(optimized_responses[i])), 'r--', 
                     label='Optimized Method')
        
        # Highlight the passband
        plt.axvspan(f_low, f_high, alpha=0.1, color='green')
        
        plt.title(f'Band {i+1}: {f_low/1e6:.1f}-{f_high/1e6:.1f} MHz Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.ylim(-80, 5)
        
        # Plot phase response
        plt.subplot(len(bandwidths), 2, i*2+2)
        
        # Plot original method
        plt.semilogx(w, np.unwrap(np.angle(original_responses[i])) * 180 / np.pi, 'b-', 
                     label='Original Method')
        
        # Plot optimized method
        plt.semilogx(w, np.unwrap(np.angle(optimized_responses[i])) * 180 / np.pi, 'r--', 
                     label='Optimized Method')
        
        # Highlight the passband
        plt.axvspan(f_low, f_high, alpha=0.1, color='green')
        
        plt.title(f'Band {i+1}: {f_low/1e6:.1f}-{f_high/1e6:.1f} MHz Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (degrees)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_filter_characteristics(fs, bandwidths, transition_width_percent=0.1):
    """
    Analyze and display key characteristics of the filters.
    
    Parameters:
    - fs: Sampling frequency in Hz
    - bandwidths: List of tuples containing low and high cutoff frequencies in Hz
    - transition_width_percent: Percentage of the passband width used for transition
    """
    nyq = fs / 2
    
    # Create results table
    print("\nFilter Characteristics Analysis:")
    print("="*80)
    print(f"{'Band':^10} | {'Freq Range (MHz)':^20} | {'Order':^10} | {'Transition Width (kHz)':^25} | {'Stopband Atten. (dB)':^20}")
    print("-"*80)
    
    for i, (f_low, f_high) in enumerate(bandwidths):
        # Calculate filter parameters
        passband_width = f_high - f_low
        transition_width = transition_width_percent * passband_width
        delta_f = transition_width / fs
        
        # Estimate filter order
        N = int(np.ceil(3.3 / delta_f))
        if N % 2 == 0:
            N += 1
        
        # Design filter
        coefficients = signal.firwin(N, [f_low / nyq, f_high / nyq], pass_zero=False, window='hamming')
        
        # Compute frequency response at high resolution
        w, h = signal.freqz(coefficients, worN=8000)
        
        # Convert frequencies to actual Hz
        freqs = w * fs / (2 * np.pi)
        
        # Find stopband attenuation (minimum beyond transition bands)
        # Define stopband regions
        lower_stop = freqs < (f_low - transition_width/2)
        upper_stop = freqs > (f_high + transition_width/2)
        stopband = np.logical_or(lower_stop, upper_stop)
        
        if np.any(stopband):
            stopband_atten = -np.max(20 * np.log10(np.abs(h[stopband])))
        else:
            stopband_atten = float('nan')
        
        # Print results
        band_range = f"{f_low/1e6:.2f}-{f_high/1e6:.2f}"
        print(f"{i+1:^10} | {band_range:^20} | {N:^10} | {transition_width/1e3:^25.2f} | {stopband_atten:^20.2f}")
    
    print("="*80)

def compare_filter_outputs(test_signal, fs, bandwidths, transition_width_percent=0.1):
    """
    Compare the output of original and optimized filter implementations.
    
    Parameters:
    - test_signal: Input signal to filter
    - fs: Sampling frequency in Hz
    - bandwidths: List of tuples containing low and high cutoff frequencies in Hz
    - transition_width_percent: Percentage of the passband width used for transition
    """
    # Get filtered signals from both implementations
    original_filtered = original_fir_hamming_filters(test_signal, fs, bandwidths, transition_width_percent=transition_width_percent)
    optimized_filtered = optimized_fir_hamming_filters(test_signal, fs, bandwidths, transition_width_percent=transition_width_percent)
    
    # Print comparison results
    print("\nFilter Output Comparison:")
    print("="*80)
    print(f"{'Band':^10} | {'Correlation':^15} | {'Mean Diff':^15} | {'Max Diff':^15} | {'RMS Diff':^15}")
    print("-"*80)
    
    for i in range(len(bandwidths)):
        # Calculate correlation
        correlation = np.abs(np.corrcoef(np.real(original_filtered[i]), np.real(optimized_filtered[i]))[0, 1])
        
        # Calculate differences
        diff = np.real(original_filtered[i]) - np.real(optimized_filtered[i])
        mean_diff = np.mean(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff**2))
        
        # Print results
        band_range = f"{bandwidths[i][0]/1e6:.2f}-{bandwidths[i][1]/1e6:.2f} MHz"
        print(f"{i+1:^10} | {correlation:^15.5f} | {mean_diff:^15.5f} | {max_diff:^15.5f} | {rms_diff:^15.5f}")
    
    print("="*80)

def performance_test(test_signal, fs, bandwidths, transition_width_percent=0.1, num_iterations=3):
    """
    Perform a detailed performance test comparing original and optimized implementations.
    
    Parameters:
    - test_signal: Input signal to filter
    - fs: Sampling frequency in Hz
    - bandwidths: List of tuples containing low and high cutoff frequencies in Hz
    - transition_width_percent: Percentage of the passband width used for transition
    - num_iterations: Number of iterations for timing (to get a more stable average)
    """
    # Prepare signals of different lengths
    sizes = [len(test_signal), len(test_signal)*10, len(test_signal)*100]
    labels = ["Small", "Medium", "Large"]
    
    # Print header
    print("\nPerformance Comparison:")
    print("="*100)
    print(f"{'Signal Size':^15} | {'Original (s)':^15} | {'Optimized (s)':^15} | {'Speedup':^10} | {'Samples':^15} | {'Avg Filter Order':^15}")
    print("-"*100)
    
    # Calculate average filter order
    filter_orders = []
    for f_low, f_high in bandwidths:
        passband_width = f_high - f_low
        transition_width = transition_width_percent * passband_width
        delta_f = transition_width / fs
        N = int(np.ceil(3.3 / delta_f))
        if N % 2 == 0:
            N += 1
        filter_orders.append(N)
    avg_order = sum(filter_orders) / len(filter_orders)
    
    # Test each signal size
    for i, size_multiplier in enumerate(sizes):
        if i == 0:
            test_data = test_signal
        else:
            # Create a larger test signal by repeating the original
            repeats = size_multiplier // len(test_signal)
            test_data = np.tile(test_signal, repeats)
        
        # Measure execution time for both implementations (multiple iterations)
        orig_times = []
        opt_times = []
        
        for _ in range(num_iterations):
            # Original implementation
            start_time = time.time()
            _ = original_fir_hamming_filters(test_data, fs, bandwidths, transition_width_percent=transition_width_percent)
            orig_times.append(time.time() - start_time)
            
            # Optimized implementation
            start_time = time.time()
            _ = optimized_fir_hamming_filters(test_data, fs, bandwidths, transition_width_percent=transition_width_percent)
            opt_times.append(time.time() - start_time)
        
        # Calculate average times
        orig_time = sum(orig_times) / len(orig_times)
        opt_time = sum(opt_times) / len(opt_times)
        
        # Calculate speedup
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        
        # Print results
        print(f"{labels[i]:^15} | {orig_time:^15.4f} | {opt_time:^15.4f} | {speedup:^10.2f}x | {len(test_data):^15} | {avg_order:^15.1f}")
    
    print("="*100)
    
    # Print theoretical analysis
    print("\nTheoretical Complexity Analysis:")
    print(f"Original implementation: O(n * k) where n is signal length and k is filter order")
    print(f"Optimized implementation: O(n log n) where n is signal length")
    print(f"Average filter order: {avg_order:.1f}")
    print(f"For large signals, the optimized implementation is theoretically more efficient")
    print(f"when n log n < n * k, which is true for n > {avg_order/np.log2(avg_order):.1f}")

# Run the tests
if __name__ == '__main__':
    # Create test instance
    test = TestOptimizedFIRHammingFilters()
    test.setUp()
    
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Create Bode diagram for the filters
    create_bode_diagram(test.fs, test.bandwidths, test.transition_width_percent)
    
    # Analyze filter characteristics
    analyze_filter_characteristics(test.fs, test.bandwidths, test.transition_width_percent)
    
    # Compare filter outputs
    compare_filter_outputs(test.test_signal, test.fs, test.bandwidths, test.transition_width_percent)
    
    # Visualize the filter performance (comparison)
    visualize_filter_performance(test.test_signal, test.fs, test.bandwidths, test.transition_width_percent)
    
    # Performance test
    performance_test(test.test_signal, test.fs, test.bandwidths, test.transition_width_percent)