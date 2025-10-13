import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.linalg import hankel, svd, eig
from scipy.interpolate import interp1d
import pywt
import h5py
import time
import pandas as pd
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['lines.linewidth'] = 1

# ===========================================
# ALGORITHM FUNCTIONS
# ===========================================

def extract_phase_signal(file_path, fs_desired=300, duration_desired=60):
    """Extract phase signal from .h5 file using variation trend method"""
    with h5py.File(file_path, "r") as f:
        frame = f["sessions/session_0/group_0/entry_0/result/frame"]
        real_part = np.array(frame["real"], dtype=np.float64)
        imag_part = np.array(frame["imag"], dtype=np.float64)

    IQ_data = real_part + 1j * imag_part
    IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

    num_sweeps = IQ_data.shape[2]
    original_duration = 60
    original_fs = num_sweeps / original_duration

    # Range bin selection
    magnitude_data = np.abs(IQ_data)
    mean_magnitude = np.mean(magnitude_data, axis=2)
    peak_range_index = np.argmax(mean_magnitude, axis=1)

    range_start_bin = max(0, peak_range_index[0] - 5)
    range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
    range_indices = np.arange(range_start_bin, range_end_bin + 1)

    # Temporal low-pass filtering
    D = 100
    tau_iq = 0.04
    f_low = 0.2

    downsampled_data = IQ_data[:, range_indices[::D], :]
    alpha_iq = np.exp(-2 / (tau_iq * original_fs))

    filtered_data = np.zeros_like(downsampled_data)
    filtered_data[:, :, 0] = downsampled_data[:, :, 0]

    for s in range(1, downsampled_data.shape[2]):
        filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                                 (1 - alpha_iq) * downsampled_data[:, :, s]

    # Phase extraction
    alpha_phi = np.exp(-2 * f_low / original_fs)
    phi_original = np.zeros(filtered_data.shape[2])

    for s in range(1, filtered_data.shape[2]):
        z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
        phi_original[s] = alpha_phi * phi_original[s - 1] + np.angle(z)

    # Resample to desired fs and duration
    original_t = np.linspace(0, original_duration, len(phi_original), endpoint=False)
    phase_interp = interp1d(original_t, phi_original, kind='cubic', fill_value="extrapolate")
    
    new_num_samples = int(duration_desired * fs_desired)
    new_t = np.linspace(0, duration_desired, new_num_samples, endpoint=False)
    phi_clean = phase_interp(new_t)

    # Add Gaussian noise
    np.random.seed(42)
    noise = 0.0 * np.random.randn(len(phi_clean))
    phi = phi_clean + noise

    return phi, fs_desired, duration_desired, new_t

def hankel_dmd(X, Y, rank=None, tol=1e-10):
    """Perform Hankel-DMD with proper eigenvalue conversion"""
    U, s, Vh = svd(X, full_matrices=False)
    
    if rank is None:
        rank = np.sum(s > tol * s[0])
    
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vh_r = Vh[:rank, :]
    
    A_tilde = U_r.conj().T @ Y @ Vh_r.conj().T @ np.diag(1/s_r)
    evals, evecs = eig(A_tilde)
    modes = Y @ Vh_r.conj().T @ np.diag(1/s_r) @ evecs
    
    dt = 1/300  # Fixed sampling rate
    omega = np.log(evals)/dt
    frequencies = np.imag(omega)/(2*np.pi)
    damping_rates = np.real(omega)
    
    return modes, evals, frequencies, damping_rates

def diagonal_averaging_improved(H_matrix):
    """Convert Hankel matrix back to 1D signal with proper handling"""
    m, n = H_matrix.shape
    L = m + n - 1
    result = np.zeros(L)
    
    for k in range(L):
        elements = []
        for i in range(max(0, k-n+1), min(m, k+1)):
            j = k - i
            if 0 <= j < n:
                elements.append(H_matrix[i, j])
        
        if elements:
            result[k] = np.mean(elements)
    
    return result

def HDMD_algorithm(phi):
    """HDMD algorithm implementation"""
    start_time = time.time()
    
    # Build Hankel matrix
    N = len(phi)
    m = 500
    n = N - m + 1
    H = hankel(phi[:n], phi[n-1:])
    
    X = H[:, :-1]
    Y = H[:, 1:]
    
    modes, evals, frequencies, damping_rates = hankel_dmd(X, Y)
    
    # Select respiratory modes
    respiratory_mask = (np.abs(frequencies) >= 0.1) & (np.abs(frequencies) < 0.8)
    respiratory_indices = np.where(respiratory_mask)[0]
    
    if len(respiratory_indices) > 0:
        sorted_respiratory = respiratory_indices[np.argsort(np.abs(evals[respiratory_indices]))[::-1]]
        n_resp_modes = min(3, len(sorted_respiratory))
        selected_resp_modes = sorted_respiratory[:n_resp_modes]
        dominant_freq_idx = selected_resp_modes[0]
        breathing_rate_hz = np.abs(frequencies[dominant_freq_idx])
        breathing_rate_bpm = breathing_rate_hz * 60

    else:
        breathing_rate_hz = 0.2
        breathing_rate_bpm = breathing_rate_hz * 60

    
    execution_time = time.time() - start_time
    return breathing_rate_bpm, execution_time

def EMD_algorithm(phi):
    """EMD algorithm implementation"""
    start_time = time.time()
    
    def emd_simple(signal, max_imfs=10):
        imfs = []
        residue = signal.copy()
        
        for i in range(max_imfs):
            imf = extract_imf(residue)
            if np.sum(np.abs(imf)) < 1e-6 * np.sum(np.abs(signal)):
                break
            imfs.append(imf)
            residue = residue - imf
            if is_monotonic(residue):
                break
        
        imfs.append(residue)
        return np.array(imfs)

    def extract_imf(signal, max_iterations=10):
        h = signal.copy()
        for _ in range(max_iterations):
            maxima_idx = find_peaks(h)
            minima_idx = find_peaks(-h)
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            upper_env = interpolate_envelope(np.arange(len(h)), h, maxima_idx)
            lower_env = interpolate_envelope(np.arange(len(h)), h, minima_idx)
            mean_env = (upper_env + lower_env) / 2
            h_new = h - mean_env
            if np.sum((h - h_new)**2) / np.sum(h**2) < 0.01:
                break
            h = h_new
        return h

    def find_peaks(signal):
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return np.array(peaks)

    def interpolate_envelope(x, signal, peak_indices):
        if len(peak_indices) < 2:
            return np.zeros_like(signal)
        extended_indices = np.concatenate([[0], peak_indices, [len(signal)-1]])
        extended_values = np.concatenate([[signal[0]], signal[peak_indices], [signal[-1]]])
        return np.interp(x, extended_indices, extended_values)

    def is_monotonic(signal):
        diff = np.diff(signal)
        return np.all(diff >= 0) or np.all(diff <= 0)

    imfs_emd = emd_simple(phi, max_imfs=10)
    
    # Analyze IMFs for breathing component
    breathing_freq_range = [0.1, 0.8]
    imf_frequencies = []
    
    for i, imf in enumerate(imfs_emd[:-1]):
        zero_crossings = np.where(np.diff(np.sign(imf)))[0]
        if len(zero_crossings) > 1:
            avg_period = 2 * (zero_crossings[-1] - zero_crossings[0]) / (len(zero_crossings) - 1)
            dominant_freq = 300 / avg_period
        else:
            freqs, psd = welch(imf, fs=300, nperseg=min(1024, len(imf)//4))
            dominant_freq = freqs[np.argmax(psd)]
        imf_frequencies.append(dominant_freq)
    
    # Find breathing frequency
    breathing_imfs = [i for i, freq in enumerate(imf_frequencies) if breathing_freq_range[0] <= freq <= breathing_freq_range[1]]
    
    if breathing_imfs:
        breathing_rate_hz = imf_frequencies[breathing_imfs[0]]
        breathing_rate_bpm = breathing_rate_hz * 60
    else:
        target_freq = 0.3
        freq_diffs = [abs(freq - target_freq) for freq in imf_frequencies]
        closest_imf = np.argmin(freq_diffs)
        breathing_rate_hz = imf_frequencies[closest_imf]
        breathing_rate_bpm = breathing_rate_hz * 60

    execution_time = time.time() - start_time
    return breathing_rate_bpm, execution_time

def VMD_algorithm(phi):
    """VMD algorithm implementation"""
    start_time = time.time()
    
    def vmd_improved(signal, K=6, alpha=2000, tau=0, init=1, tol=1e-6, max_iter=200, fs=300):
        T = len(signal)
        f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])
        N = len(f_mirror)
        freqs = np.fft.fftfreq(N, d=1/fs)
        freqs = np.fft.fftshift(freqs)
        
        omega = np.zeros(K)
        if init == 1:
            omega = np.linspace(0.05, 0.45, K)
        elif init == 2:
            omega[0] = 0.05
            omega[1] = 0.20
            omega[2] = 0.25
            omega[3] = 0.30
            if K > 4:
                omega[4] = 0.35
                omega[5] = 0.40
        
        u_hat = np.zeros((K, N), dtype=complex)
        omega_old = omega.copy()
        lambda_hat = np.zeros(N, dtype=complex)
        f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
        
        n_iter = 0
        eps = 1e-9
        
        while n_iter < max_iter:
            for k in range(K):
                sum_uk = np.sum(u_hat, axis=0) - u_hat[k, :]
                numerator = f_hat - sum_uk - lambda_hat / 2
                denominator = 1 + alpha * (freqs - omega[k])**2
                u_hat[k, :] = numerator / (denominator + eps)
                power_spectrum = np.abs(u_hat[k, :])**2
                omega[k] = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + eps)
                omega[k] = np.clip(omega[k], 0, fs/2)
            
            residual = f_hat - np.sum(u_hat, axis=0)
            lambda_hat = lambda_hat + tau * residual
            
            if n_iter > 0:
                omega_diff = np.sum(np.abs(omega - omega_old))
                if omega_diff < tol:
                    break
            
            omega_old = omega.copy()
            n_iter += 1
        
        u = np.zeros((K, T))
        for k in range(K):
            u_temp = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :])))
            u[k, :] = u_temp[T:2*T]
        
        return u, omega
    
    modes_vmd, omega_vmd = vmd_improved(phi, K=6, alpha=2000, tau=0, init=2, fs=300.0)
    
    # Select breathing modes
    breathing_modes_vmd = [i for i, f in enumerate(omega_vmd) if 0.1 <= f <= 0.8]
    
    if breathing_modes_vmd:
        mode_energies = [np.sum(np.abs(modes_vmd[i, :])**2) for i in breathing_modes_vmd]
        dominant_mode_idx = breathing_modes_vmd[np.argmax(mode_energies)]
        breathing_rate_hz = omega_vmd[dominant_mode_idx]
        breathing_rate_bpm = breathing_rate_hz * 60

    else:
        target_freq = 0.3
        freq_diffs = [abs(freq - target_freq) for freq in omega_vmd]
        dominant_mode_idx = np.argmin(freq_diffs)
        breathing_rate_hz = omega_vmd[dominant_mode_idx]
        breathing_rate_bpm = breathing_rate_hz * 60
    
    execution_time = time.time() - start_time
    return breathing_rate_bpm, execution_time

def DWT_algorithm(phi):
    """DWT algorithm implementation"""
    start_time = time.time()
    
    def analyze_wavelet_breathing(signal, fs=300, wavelet='db4', max_level=4):
        optimal_level = min(max_level, int(np.log2(len(signal))))
        coeffs = pywt.wavedec(signal, wavelet, level=optimal_level, mode='periodization')
        coeff_names = [f'A{optimal_level}'] + [f'D{i}' for i in range(optimal_level, 0, -1)]
        
        best_breathing_rate = 0
        max_resp_power = 0
        
        for i, (coeff, name) in enumerate(zip(coeffs, coeff_names)):
            if coeff is None or len(coeff) < 10:
                continue
                
            if name.startswith('A'):
                level = int(name[1:])
                effective_fs = fs / (2 ** level)
            else:
                level = int(name[1:])
                effective_fs = fs / (2 ** (level - 1))
            
            test_coeffs = [np.zeros_like(c) if j != i else c for j, c in enumerate(coeffs)]
            reconstructed = pywt.waverec(test_coeffs, wavelet, mode='periodization')
            
            if len(reconstructed) > len(signal):
                reconstructed = reconstructed[:len(signal)]
            elif len(reconstructed) < len(signal):
                reconstructed = np.pad(reconstructed, (0, len(signal) - len(reconstructed)))
            
            fft_vals = np.fft.fft(reconstructed)
            fft_freqs = np.fft.fftfreq(len(reconstructed), d=1/fs)
            positive_mask = fft_freqs >= 0
            pos_freqs = fft_freqs[positive_mask]
            pos_mag = np.abs(fft_vals[positive_mask])
            
            resp_mask = (pos_freqs >= 0.04) & (pos_freqs <= 0.6)
            resp_freqs = pos_freqs[resp_mask]
            resp_mag = pos_mag[resp_mask]
            
            if len(resp_mag) > 0:
                dominant_freq = resp_freqs[np.argmax(resp_mag)]
                resp_power = np.max(resp_mag)
                
                if resp_power > max_resp_power:
                    max_resp_power = resp_power
                    best_breathing_rate = dominant_freq
        
        if best_breathing_rate == 0 and f'A{optimal_level}' in dict(zip(coeff_names, coeffs)):
            best_breathing_rate = 0.2
        
        return best_breathing_rate
    
    breathing_rate_hz = analyze_wavelet_breathing(phi, fs=300)
    breathing_rate_bpm = breathing_rate_hz * 60

    execution_time = time.time() - start_time
    return breathing_rate_bpm, execution_time

# ===========================================
# MAIN PROCESSING LOOP
# ===========================================

def main():
    # Define ICU TEST folder path
    icu_test_folder = r"C:\Users\GOPAL\neurips dataset\ICU TEST"
    
    # Initialize results storage
    results = []
    
    # Process each file from i1.h5 to i24.h5
    for i in range(1, 25):
        print(f"\n{'='*50}")
        print(f"PROCESSING FILE i{i}.h5")
        print(f"{'='*50}")
        
        # Initialize row for this index
        row = {'Index': i}
        
        try:
            # Construct file path
            file_path = os.path.join(icu_test_folder, f"i{i}.h5")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                # Add empty row with index only
                empty_row = {'Index': i}
                for algo in ['HDMD', 'EMD', 'VMD', 'DWT']:
                    empty_row[algo] = np.nan
                    empty_row[f'{algo}_time'] = np.nan
                results.append(empty_row)
                continue
            
            # Extract phase signal
            phi, _, _, _ = extract_phase_signal(file_path)
            
            # Process with HDMD
            print("Running HDMD...")
            hdmd_bpm, hdmd_time = HDMD_algorithm(phi)
            row.update({
                'HDMD': hdmd_bpm,
                'HDMD_time': hdmd_time
            })
            
            # Process with EMD
            print("Running EMD...")
            emd_bpm, emd_time = EMD_algorithm(phi)
            row.update({
                'EMD': emd_bpm,
                'EMD_time': emd_time
            })
            
            # Process with VMD
            print("Running VMD...")
            vmd_bpm, vmd_time = VMD_algorithm(phi)
            row.update({
                'VMD': vmd_bpm,
                'VMD_time': vmd_time
            })
            
            # Process with DWT
            print("Running DWT...")
            dwt_bpm, dwt_time = DWT_algorithm(phi)
            row.update({
                'DWT': dwt_bpm,
                'DWT_time': dwt_time
            })
            
            # Add row to results
            results.append(row)
            print(f"Completed file i{i}.h5")
            
        except Exception as e:
            print(f"Error processing file i{i}.h5: {str(e)}")
            # Add empty row with index only
            empty_row = {'Index': i}
            for algo in ['HDMD', 'EMD', 'VMD', 'DWT']:
                empty_row[algo] = np.nan
                empty_row[f'{algo}_time'] = np.nan
            results.append(empty_row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Reorder columns to match required format
    columns_order = ['Index', 'HDMD', 'HDMD_time', 'EMD', 'EMD_time', 'VMD', 'VMD_time', 'DWT', 'DWT_time']
    df = df[columns_order]
    
    # Save to CSV
    output_file = "icu_test_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Processed {len(results)} files")
    print(f"Output file: {output_file}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows: {len(df)}")
    
    # Display first few rows
    print(f"\nFirst 5 rows of results:")
    print(df.head())

if __name__ == "__main__":
    main()