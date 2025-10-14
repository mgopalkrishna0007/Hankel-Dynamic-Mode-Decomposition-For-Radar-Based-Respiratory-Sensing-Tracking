import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.linalg import hankel, svd, eig
from scipy.interpolate import interp1d
import pywt
import h5py
import time
import os
import pandas as pd
from PyEMD import EEMD

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['lines.linewidth'] = 1

# ===========================================
# VARIATION TREND METHOD FOR PHASE EXTRACTION
# ===========================================

def extract_phase_signal(file_path, fs_desired=300, duration_desired=60):
    """Extract phase signal from radar data file"""
    try:
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

        # Phase extraction (variation trend method)
        alpha_phi = np.exp(-2 * f_low / original_fs)
        phi_original = np.zeros(filtered_data.shape[2])

        for s in range(1, filtered_data.shape[2]):
            z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
            phi_original[s] = alpha_phi * phi_original[s - 1] + np.angle(z)

        # Original time vector
        original_t = np.linspace(0, original_duration, len(phi_original), endpoint=False)

        # Resample to desired fs and duration
        phase_interp = interp1d(original_t, phi_original, kind='cubic', fill_value="extrapolate")
        new_num_samples = int(duration_desired * fs_desired)
        new_t = np.linspace(0, duration_desired, new_num_samples, endpoint=False)
        phi_clean = phase_interp(new_t)

        # Add Gaussian noise
        np.random.seed(42)
        noise = 0.0 * np.random.randn(len(phi_clean))
        phi = phi_clean + noise

        return phi, fs_desired, duration_desired, new_t
    except Exception as e:
        print(f"Error extracting phase signal from {file_path}: {e}")
        return None, None, None, None

# =============================================
# DECOMPOSITION METHODS
# =============================================

def hankel_dmd(X, Y, fs, rank=None, tol=1e-10):
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
    
    dt = 1/fs
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

def reconstruct_from_modes(selected_modes, X, modes, evals, fs):
    modes_selected = modes[:, selected_modes]
    b = np.linalg.pinv(modes_selected) @ X[:, 0]
    
    dt = 1/fs
    omega_selected = np.log(evals[selected_modes]) / dt
    time_steps = np.arange(X.shape[1])
    
    time_dynamics = np.zeros((len(selected_modes), len(time_steps)), dtype=complex)
    for i, idx in enumerate(selected_modes):
        time_dynamics[i, :] = b[i] * np.exp(omega_selected[i] * time_steps * dt)
    
    X_dmd = np.real(modes_selected @ time_dynamics)
    reconstructed = diagonal_averaging_improved(X_dmd)
    return reconstructed

def extract_respiration_eemd(signal, fs):
    """Apply EEMD and extract the dominant respiration-related IMF"""
    eemd = EEMD()
    eemd.noise_width = 0.2
    eemd.trials = 100
    eemd.noise_seed(123)

    imfs = eemd.eemd(signal)

    breathing_freq_range = [0.1, 0.8]
    imf_freqs = []
    imf_powers = []

    for i, imf in enumerate(imfs):
        nperseg = max(256, len(imf)//2)
        freqs, psd = welch(imf, fs=fs, nperseg=nperseg)

        if len(freqs) == 0 or np.all(psd == 0):
            imf_freqs.append(0)
            imf_powers.append(0)
            continue

        dominant_freq = freqs[np.argmax(psd)]
        imf_freqs.append(dominant_freq)
        
        mask = (freqs >= breathing_freq_range[0]) & (freqs <= breathing_freq_range[1])
        imf_power = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0
        imf_powers.append(imf_power)

    imf_freqs = np.array(imf_freqs)
    imf_powers = np.array(imf_powers)
    target_freq = 0.3
    score = imf_powers * np.exp(-((imf_freqs - target_freq) ** 2) / 0.1)
    dominant_imf_index = np.argmax(score)

    respiration_signal = imfs[dominant_imf_index]

    return respiration_signal, imfs, dominant_imf_index, imf_freqs

def vmd_improved(signal, K=6, alpha=2000, tau=0, DC=False, init=1, tol=1e-6, max_iter=200, fs=300):
    """Improved Variational Mode Decomposition implementation"""
    T = len(signal)
    f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])
    
    N = len(f_mirror)
    t = np.arange(N)
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

def analyze_wavelet_breathing(signal, fs, wavelet='db4', max_level=4):
    """Traditional DWT-based breathing component extraction with FFT frequency estimation"""
    optimal_level = min(max_level, int(np.log2(len(signal))))
    coeffs = pywt.wavedec(signal, wavelet, level=optimal_level, mode='periodization')
    coeff_names = [f'A{optimal_level}'] + [f'D{i}' for i in range(optimal_level, 0, -1)]
    
    best_breathing_component = None
    best_breathing_rate = 0
    best_component_name = ""
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
            dominant_idx = np.argmax(resp_mag)
            dominant_freq = resp_freqs[dominant_idx]
            resp_power = np.max(resp_mag)
            
            if resp_power > max_resp_power:
                max_resp_power = resp_power
                best_breathing_component = reconstructed
                best_breathing_rate = dominant_freq
                best_component_name = name
    
    if best_breathing_component is None and f'A{optimal_level}' in dict(zip(coeff_names, coeffs)):
        a4_idx = coeff_names.index(f'A{optimal_level}')
        test_coeffs = [np.zeros_like(c) if j != a4_idx else c for j, c in enumerate(coeffs)]
        best_breathing_component = pywt.waverec(test_coeffs, wavelet, mode='periodization')
        best_component_name = f'A{optimal_level}'
        
        fft_vals = np.fft.fft(best_breathing_component)
        fft_freqs = np.fft.fftfreq(len(best_breathing_component), d=1/fs)
        resp_mask = (fft_freqs >= 0.04) & (fft_freqs <= 0.6) & (fft_freqs >= 0)
        if np.any(resp_mask):
            resp_freqs = fft_freqs[resp_mask]
            resp_mag = np.abs(fft_vals[resp_mask])
            best_breathing_rate = resp_freqs[np.argmax(resp_mag)]
        else:
            best_breathing_rate = 0.2
    
    return best_breathing_component, best_component_name, best_breathing_rate

def process_single_file(file_path):
    """Process a single file and return all decomposition results"""
    # Extract phase signal
    phi, fs, duration, t = extract_phase_signal(file_path)
    if phi is None:
        return None
    
    results = {}
    
    # 1. HDMD
    try:
        N = len(phi)
        m = 500
        n = N - m + 1
        H = hankel(phi[:n], phi[n-1:])
        X = H[:, :-1]
        Y = H[:, 1:]
        
        start_time = time.time()
        modes, evals, frequencies, damping_rates = hankel_dmd(X, Y, fs)
        hdmd_time = time.time() - start_time
        
        respiratory_mask = (np.abs(frequencies) >= 0.1) & (np.abs(frequencies) < 0.8)
        respiratory_indices = np.where(respiratory_mask)[0]
        
        if len(respiratory_indices) > 0:
            sorted_respiratory = respiratory_indices[np.argsort(np.abs(evals[respiratory_indices]))[::-1]]
            n_resp_modes = min(3, len(sorted_respiratory))
            selected_resp_modes = sorted_respiratory[:n_resp_modes]
            dominant_freq_idx = selected_resp_modes[0]
            breathing_rate_hdmd_hz = np.abs(frequencies[dominant_freq_idx])
        else:
            breathing_rate_hdmd_hz = 0.2
        
        breathing_rate_hdmd_bpm = breathing_rate_hdmd_hz * 60
        results['hdmd_bpm'] = breathing_rate_hdmd_bpm
        results['hdmd_time'] = hdmd_time
    except Exception as e:
        print(f"HDMD failed for {file_path}: {e}")
        results['hdmd_bpm'] = 0
        results['hdmd_time'] = 0
    
    # 2. EEMD
    try:
        start_time = time.time()
        breathing_emd, imfs_emd, dominant_imf_idx, imf_frequencies = extract_respiration_eemd(phi, fs)
        emd_time = time.time() - start_time
        
        freqs, psd = welch(breathing_emd, fs=fs, nperseg=max(256, len(breathing_emd)//2))
        dominant_freq_idx = np.argmax(psd)
        breathing_rate_emd_hz = freqs[dominant_freq_idx]
        
        if not (0.1 <= breathing_rate_emd_hz <= 0.8):
            breathing_rate_emd_hz = 0.3
        
        breathing_rate_emd_bpm = breathing_rate_emd_hz * 60
        results['eemd_bpm'] = breathing_rate_emd_bpm
        results['eemd_time'] = emd_time
    except Exception as e:
        print(f"EEMD failed for {file_path}: {e}")
        results['eemd_bpm'] = 0
        results['eemd_time'] = 0
    
    # 3. VMD
    try:
        start_time = time.time()
        modes_vmd, omega_vmd = vmd_improved(phi, K=6, alpha=2000, tau=0, init=2, fs=300.0)
        vmd_time = time.time() - start_time
        
        breathing_modes_vmd = [i for i, f in enumerate(omega_vmd) if 0.1 <= f <= 0.8]
        
        if not breathing_modes_vmd:
            target_freq = 0.3
            freq_diffs = [abs(freq - target_freq) for freq in omega_vmd]
            dominant_mode_idx = np.argmin(freq_diffs)
        else:
            mode_energies = [np.sum(np.abs(modes_vmd[i, :])**2) for i in breathing_modes_vmd]
            dominant_mode_idx = breathing_modes_vmd[np.argmax(mode_energies)]
        
        breathing_rate_vmd_hz = omega_vmd[dominant_mode_idx]
        breathing_rate_vmd_bpm = breathing_rate_vmd_hz * 60
        results['vmd_bpm'] = breathing_rate_vmd_bpm
        results['vmd_time'] = vmd_time
    except Exception as e:
        print(f"VMD failed for {file_path}: {e}")
        results['vmd_bpm'] = 0
        results['vmd_time'] = 0
    
    # 4. Wavelet
    try:
        start_time = time.time()
        breathing_wavelet, wavelet_level, breathing_rate_wavelet_hz = analyze_wavelet_breathing(phi, fs, wavelet='db4', max_level=4)
        wavelet_time = time.time() - start_time
        
        breathing_rate_wavelet_bpm = breathing_rate_wavelet_hz * 60
        results['dwt_bpm'] = breathing_rate_wavelet_bpm
        results['dwt_time'] = wavelet_time
    except Exception as e:
        print(f"Wavelet failed for {file_path}: {e}")
        results['dwt_bpm'] = 0
        results['dwt_time'] = 0
    
    return results

# =============================================
# MAIN PROCESSING LOOP
# =============================================

def main():
    # Define folders and file patterns
    folders = {
        'no_lens': r"C:\Users\GOPAL\neurips dataset\NO LENS",
        'convex': r"C:\Users\GOPAL\neurips dataset\CONV", 
        'plane_lens': r"C:\Users\GOPAL\neurips dataset\PlANE LENS"
    }
    
    # Initialize results dictionary
    all_results = {
        'hdmd_no_lens': [], 'hdmd_convex': [], 'hdmd_plane_lens': [],
        'eemd_no_lens': [], 'eemd_convex': [], 'eemd_plane_lens': [],
        'vmd_no_lens': [], 'vmd_convex': [], 'vmd_plane_lens': [],
        'dwt_no_lens': [], 'dwt_convex': [], 'dwt_plane_lens': [],
        'hdmd_time': [], 'eemd_time': [], 'vmd_time': [], 'dwt_time': []
    }
    
    # Process files 1 to 24 for each folder
    for i in range(1, 25):
        print(f"\n{'='*50}")
        print(f"PROCESSING FILE SET {i}")
        print(f"{'='*50}")
        
        file_results = {}
        
        for lens_type, folder_path in folders.items():
            if lens_type == 'no_lens':
                filename = f"nolens{i}.h5"
            elif lens_type == 'convex':
                filename = f"conv{i}.h5"
            elif lens_type == 'plane_lens':
                filename = f"pn{i}.h5"
            
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")
            
            if os.path.exists(file_path):
                results = process_single_file(file_path)
                if results:
                    file_results[lens_type] = results
                else:
                    # Use default values if processing failed
                    file_results[lens_type] = {
                        'hdmd_bpm': 0, 'hdmd_time': 0,
                        'eemd_bpm': 0, 'eemd_time': 0,
                        'vmd_bpm': 0, 'vmd_time': 0,
                        'dwt_bpm': 0, 'dwt_time': 0
                    }
            else:
                print(f"File not found: {file_path}")
                file_results[lens_type] = {
                    'hdmd_bpm': 0, 'hdmd_time': 0,
                    'eemd_bpm': 0, 'eemd_time': 0,
                    'vmd_bpm': 0, 'vmd_time': 0,
                    'dwt_bpm': 0, 'dwt_time': 0
                }
        
        # Store results for this file set
        for lens_type in ['no_lens', 'convex', 'plane_lens']:
            if lens_type in file_results:
                results = file_results[lens_type]
                all_results[f'hdmd_{lens_type}'].append(results.get('hdmd_bpm', 0))
                all_results[f'eemd_{lens_type}'].append(results.get('eemd_bpm', 0))
                all_results[f'vmd_{lens_type}'].append(results.get('vmd_bpm', 0))
                all_results[f'dwt_{lens_type}'].append(results.get('dwt_bpm', 0))
        
        # Calculate average times across all lens types
        hdmd_times = [file_results[lens_type].get('hdmd_time', 0) for lens_type in file_results]
        eemd_times = [file_results[lens_type].get('eemd_time', 0) for lens_type in file_results]
        vmd_times = [file_results[lens_type].get('vmd_time', 0) for lens_type in file_results]
        dwt_times = [file_results[lens_type].get('dwt_time', 0) for lens_type in file_results]
        
        all_results['hdmd_time'].append(np.mean(hdmd_times))
        all_results['eemd_time'].append(np.mean(eemd_times))
        all_results['vmd_time'].append(np.mean(vmd_times))
        all_results['dwt_time'].append(np.mean(dwt_times))
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    df.index = range(1, 25)  # Index from 1 to 24
    
    # Reorder columns for better readability
    columns_order = [
        'hdmd_no_lens', 'hdmd_convex', 'hdmd_plane_lens',
        'eemd_no_lens', 'eemd_convex', 'eemd_plane_lens', 
        'vmd_no_lens', 'vmd_convex', 'vmd_plane_lens',
        'dwt_no_lens', 'dwt_convex', 'dwt_plane_lens',
        'hdmd_time', 'eemd_time', 'vmd_time', 'dwt_time'
    ]
    
    df = df[columns_order]
    
    # Save to CSV
    output_path = r"C:\Users\GOPAL\neurips dataset\decomposition_results.csv"
    df.to_csv(output_path, index=True, index_label='File_Number')
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"\nSummary of results:")
    print(df.head())
    
    # Print summary statistics
    print(f"\nAverage Breathing Rates (BPM):")
    for col in ['hdmd_no_lens', 'hdmd_convex', 'hdmd_plane_lens', 
                'eemd_no_lens', 'eemd_convex', 'eemd_plane_lens',
                'vmd_no_lens', 'vmd_convex', 'vmd_plane_lens',
                'dwt_no_lens', 'dwt_convex', 'dwt_plane_lens']:
        if col in df.columns:
            avg_bpm = df[col].mean()
            print(f"  {col}: {avg_bpm:.2f} BPM")
    
    print(f"\nAverage Computational Times (seconds):")
    for col in ['hdmd_time', 'eemd_time', 'vmd_time', 'dwt_time']:
        if col in df.columns:
            avg_time = df[col].mean()
            print(f"  {col}: {avg_time:.4f} s")

if __name__ == "__main__":
    main()
