import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.linalg import hankel, svd, eig
from scipy.interpolate import interp1d
import pywt
import h5py
import time

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['mathtext.fontset'] = 'stix'   # For math text (to match Times New Roman)

# Set global line width
plt.rcParams['lines.linewidth'] = 1

# ===========================================
# VARIATION TREND METHOD FOR PHASE EXTRACTION
# ===========================================
print("="*50)
print("VARIATION TREND METHOD FOR PHASE EXTRACTION")
print("="*50)

# ---------------------------
# User Parameters
# ---------------------------
fs_desired = 300  # Desired sampling frequency in Hz (input parameter)
duration_desired = 60  # Desired duration in seconds

# ---------------------------
# Load radar data
# ---------------------------
file_path = r"c:\Users\GOPAL\neurips dataset\ICU TEST\i1.h5"

with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # Real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Imag part

# Combine into complex IQ data: shape (1794, 32, 40)
IQ_data = real_part + 1j * imag_part

# Transpose to (antennas x range bins x sweeps)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# ---------------------------
# Parameters
# ---------------------------
num_sweeps = IQ_data.shape[2]  # number of time samples (frames)
original_duration = 60         # original duration in seconds
original_fs = num_sweeps / original_duration  # original sampling frequency in Hz

print(f"\nOriginal data parameters:")
print(f"  - Number of sweeps: {num_sweeps}")
print(f"  - Duration: {original_duration} s")
print(f"  - Sampling frequency: {original_fs:.2f} Hz")

print(f"\nDesired output parameters:")
print(f"  - Duration: {duration_desired} s")
print(f"  - Sampling frequency: {fs_desired} Hz")
print(f"  - Number of points: {int(duration_desired * fs_desired)}")

# ---------------------------
# Range bin selection
# ---------------------------
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)   # Average over time
peak_range_index = np.argmax(mean_magnitude, axis=1)  # Peak bin per antenna

range_start_bin = max(0, peak_range_index[0] - 5)
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

print(f"\nSelected range bins: {range_start_bin} to {range_end_bin}")

# ---------------------------
# Temporal low-pass filtering
# ---------------------------
D = 100                        # Downsampling factor (range bins)
tau_iq = 0.04                  # Low-pass filter time constant (s)
f_low = 0.2                    # High-pass filter cutoff (Hz)

downsampled_data = IQ_data[:, range_indices[::D], :]  # keep sweeps intact
alpha_iq = np.exp(-2 / (tau_iq * original_fs))

filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# ---------------------------
# Phase extraction (variation trend method)
# ---------------------------
alpha_phi = np.exp(-2 * f_low / original_fs)
phi_original = np.zeros(filtered_data.shape[2])

for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi_original[s] = alpha_phi * phi_original[s - 1] + np.angle(z)

# Original time vector
original_t = np.linspace(0, original_duration, len(phi_original), endpoint=False)

# ---------------------------
# Resample to desired fs and duration
# ---------------------------
# Create interpolation function for the phase signal
phase_interp = interp1d(original_t, phi_original, kind='cubic', fill_value="extrapolate")

# Create new time vector
new_num_samples = int(duration_desired * fs_desired)
new_t = np.linspace(0, duration_desired, new_num_samples, endpoint=False)

# Resample the phase signal
phi_clean = phase_interp(new_t)

# Add Gaussian noise to the resampled phase signal
np.random.seed(42)  # For reproducibility
noise = 0.0* np.random.randn(len(phi_clean))
phi = phi_clean + noise

print(f"\nPhase signal extracted and resampled:")
print(f"  - Original samples: {len(phi_original)}")
print(f"  - Resampled samples: {len(phi)}")
print(f"  - Final sampling rate: {fs_desired} Hz")
print(f"  - Final duration: {duration_desired} s")
print(f"  - Added Gaussian noise: 2*σ")

# Use resampled parameters for decomposition
fs = fs_desired
duration = duration_desired
t = new_t

# Plot the resampled phase signal
plt.figure(figsize=(12, 4))
plt.plot(t, phi, label='Noisy Phase Signal (φ) from Variation Trend Method', color='blue')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Phase [radians]', fontsize=12)
plt.title('Phase Signal with Gaussian Noise (2σ) - Variation Trend Method', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(False, alpha=0.3)
plt.tight_layout()
plt.show()

print("Phase signal extracted successfully!")
print(f"Signal length: {len(phi)} samples")
print(f"Duration: {duration} seconds")
print(f"Sampling rate: {fs} Hz")
print(f"Noise level: 2σ Gaussian noise added")

# =============================================
# 1. HANKEL DYNAMIC MODE DECOMPOSITION (HDMD)
# =============================================
print("\n" + "="*50)
print("1. HANKEL DYNAMIC MODE DECOMPOSITION (HDMD)")
print("="*50)

# Build Hankel matrix
N = len(phi)  # Total number of samples
m = 500  # number of rows
n = N - m + 1  # number of columns
H = hankel(phi[:n], phi[n-1:])

print(f"Hankel matrix shape: {H.shape}")

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

# Prepare data matrices
X = H[:, :-1]
Y = H[:, 1:]

# Perform HDMD with timing
start_time = time.time()
modes, evals, frequencies, damping_rates = hankel_dmd(X, Y)
hdmd_time = time.time() - start_time

# Select respiratory modes (breathing frequency range: 0.1-0.8 Hz)
respiratory_mask = (np.abs(frequencies) >= 0.1) & (np.abs(frequencies) < 0.8)
respiratory_indices = np.where(respiratory_mask)[0]

if len(respiratory_indices) > 0:
    # Sort by magnitude (most dominant first)
    sorted_respiratory = respiratory_indices[np.argsort(np.abs(evals[respiratory_indices]))[::-1]]
    n_resp_modes = min(3, len(sorted_respiratory))
    selected_resp_modes = sorted_respiratory[:n_resp_modes]
    
    # Get the dominant breathing frequency from HDMD eigenvalues
    dominant_freq_idx = selected_resp_modes[0]  # Take the most dominant mode
    breathing_rate_hdmd_hz = np.abs(frequencies[dominant_freq_idx])
    breathing_rate_hdmd_bpm = breathing_rate_hdmd_hz * 60
    
    print(f"HDMD selected {n_resp_modes} respiratory modes")
    print(f"Dominant breathing mode frequency: {breathing_rate_hdmd_hz:.3f} Hz")
else:
    print("No respiratory modes found in HDMD, using fallback")
    selected_resp_modes = [0]
    breathing_rate_hdmd_hz = 0.2
    breathing_rate_hdmd_bpm = 12

# Reconstruct breathing component using HDMD
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

breathing_hdmd = reconstruct_from_modes(selected_resp_modes, X, modes, evals, fs)

print(f"\nHDMD Results:")
print(f"  - Computational time: {hdmd_time:.4f} seconds")
print(f"  - Breathing rate: {breathing_rate_hdmd_hz:.3f} Hz ({breathing_rate_hdmd_bpm:.2f} BPM)")
print(f"  - Selected modes: {selected_resp_modes}")
print(f"  - Mode frequencies: {[f'{frequencies[i]:.3f} Hz' for i in selected_resp_modes]}")

# =============================================
# 2. EMPIRICAL MODE DECOMPOSITION (EMD)
# =============================================

print("\n" + "="*50)
print("2. ENSEMBLE EMPIRICAL MODE DECOMPOSITION (EEMD)")
print("="*50)

from PyEMD import EEMD
import numpy as np
from scipy.signal import welch
import time

# =========================================================
# Function to apply EEMD and extract the respiration signal
# =========================================================
def extract_respiration_eemd(signal, fs):
    """Apply EEMD and extract the dominant respiration-related IMF"""
    eemd = EEMD()
    imfs = eemd(signal)
    
    breathing_freq_range = [0.1, 0.8]  # Hz
    imf_freqs = []
    imf_powers = []

    # Compute dominant frequency of each IMF
    for i, imf in enumerate(imfs):
        freqs, psd = welch(imf, fs=fs, nperseg=min(1024, len(imf)//4))
        dominant_freq = freqs[np.argmax(psd)]
        imf_freqs.append(dominant_freq)
        
        # Compute power within breathing range
        mask = (freqs >= breathing_freq_range[0]) & (freqs <= breathing_freq_range[1])
        imf_power = np.trapz(psd[mask], freqs[mask])
        imf_powers.append(imf_power)

    # Find IMF with maximum power in breathing band
    dominant_imf_index = np.argmax(imf_powers)
    respiration_signal = imfs[dominant_imf_index]

    return respiration_signal, imfs, dominant_imf_index, imf_freqs

# Apply EEMD with timing
start_time = time.time()
breathing_emd, imfs_emd, dominant_imf_idx, imf_frequencies = extract_respiration_eemd(phi, fs)
emd_time = time.time() - start_time

print(f"EEMD extracted {len(imfs_emd)} IMFs")

# Analyze dominant IMF for breathing frequency
freqs, psd = welch(breathing_emd, fs=fs, nperseg=min(1024, len(breathing_emd)//4))
dominant_freq_idx = np.argmax(psd)
breathing_rate_emd_hz = freqs[dominant_freq_idx]
breathing_rate_emd_bpm = breathing_rate_emd_hz * 60

# Check breathing range validity
if not (0.1 <= breathing_rate_emd_hz <= 0.8):
    print("Dominant frequency outside typical breathing range, adjusting to 0.3 Hz")
    breathing_rate_emd_hz = 0.3
    breathing_rate_emd_bpm = 18

print(f"\nEEMD Results:")
print(f"  - Computational time: {emd_time:.4f} seconds")
print(f"  - Breathing rate: {breathing_rate_emd_hz:.3f} Hz ({breathing_rate_emd_bpm:.2f} BPM)")
print(f"  - Dominant IMF: {dominant_imf_idx+1}")
print(f"  - Dominant IMF frequency: {imf_frequencies[dominant_imf_idx]:.3f} Hz")

# =============================================
# 3. VARIATIONAL MODE DECOMPOSITION (VMD)
# =============================================
print("\n" + "="*50)
print("3. VARIATIONAL MODE DECOMPOSITION (VMD)")
print("="*50)

def vmd_improved(signal, K=6, alpha=2000, tau=0, DC=False, init=1, tol=1e-6, max_iter=200, fs = 300):
    """Improved Variational Mode Decomposition implementation"""
    # Mirror extend signal to reduce boundary effects (full mirror)
    T = len(signal)
    f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])
    
    # Time and frequency domains
    N = len(f_mirror)
    t = np.arange(N)
    freqs = np.fft.fftfreq(N, d=1/fs)          # frequency bins in Hz
    freqs = np.fft.fftshift(freqs)             # shift zero to center
    
    # Initialize center frequencies
    omega = np.zeros(K)
    if init == 1:
        # Uniform distribution
        omega = np.linspace(0.05, 0.45, K)
    elif init == 2:
        # Initialize based on expected breathing frequencies
        omega[0] = 0.05   # Low frequency component
        omega[1] = 0.20   # First breathing frequency
        omega[2] = 0.25   # Second breathing frequency
        omega[3] = 0.30   # Third breathing frequency
        if K > 4:
            omega[4] = 0.35   # Higher breathing frequency
            omega[5] = 0.40   # High frequency component
    
    # Initialize modes in frequency domain
    u_hat = np.zeros((K, N), dtype=complex)
    omega_old = omega.copy()
    
    # Lagrange multiplier
    lambda_hat = np.zeros(N, dtype=complex)
    
    # FFT of input
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    
    # Main iteration loop
    n_iter = 0
    eps = 1e-9
    
    while n_iter < max_iter:
        # Update modes
        for k in range(K):
            # Sum of all modes except k
            sum_uk = np.sum(u_hat, axis=0) - u_hat[k, :]
            
            # Update mode k
            numerator = f_hat - sum_uk - lambda_hat / 2
            denominator = 1 + alpha * (freqs - omega[k])**2
            u_hat[k, :] = numerator / (denominator + eps)
            
            # Update center frequency
            power_spectrum = np.abs(u_hat[k, :])**2
            omega[k] = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + eps)
            omega[k] = np.clip(omega[k], 0, fs/2)  # Ensure valid frequency range
        
        # Update Lagrange multiplier
        residual = f_hat - np.sum(u_hat, axis=0)
        lambda_hat = lambda_hat + tau * residual
        
        # Check convergence
        if n_iter > 0:
            omega_diff = np.sum(np.abs(omega - omega_old))
            if omega_diff < tol:
                break
        
        omega_old = omega.copy()
        n_iter += 1
    
    print(f"VMD converged after {n_iter} iterations")
    
    # Reconstruct modes in time domain
    u = np.zeros((K, T))
    for k in range(K):
        # IFFT and extract original signal length
        u_temp = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :])))
        u[k, :] = u_temp[T:2*T]  # Extract middle part matching signal length
    
    return u, omega

# Apply VMD with timing
start_time = time.time()
modes_vmd, omega_vmd = vmd_improved(phi, K=6, alpha=2000, tau=0, init=2, fs=300.0)
vmd_time = time.time() - start_time

print(f"VMD extracted {len(modes_vmd)} modes")
print(f"VMD center frequencies: {omega_vmd}")

# Select breathing modes based on VMD center frequencies
breathing_modes_vmd = [i for i, f in enumerate(omega_vmd) if 0.1 <= f <= 0.8]

# If no modes found, use the one closest to 0.3 Hz
if not breathing_modes_vmd:
    target_freq = 0.3  # Typical breathing frequency
    freq_diffs = [abs(freq - target_freq) for freq in omega_vmd]
    dominant_mode_idx = np.argmin(freq_diffs)
    print(f"No VMD mode in breathing range, selecting mode {dominant_mode_idx+1} with freq {omega_vmd[dominant_mode_idx]:.3f} Hz")
else:
    # Select the single dominant respiratory mode (highest energy)
    mode_energies = [np.sum(np.abs(modes_vmd[i, :])**2) for i in breathing_modes_vmd]
    dominant_mode_idx = breathing_modes_vmd[np.argmax(mode_energies)]

# Compute breathing rate from dominant mode
breathing_rate_vmd_hz = omega_vmd[dominant_mode_idx]
breathing_rate_vmd_bpm = breathing_rate_vmd_hz * 60

# Extract only the dominant respiratory mode
breathing_vmd = modes_vmd[dominant_mode_idx, :]

print(f"\nVMD Results:")
print(f"  - Computational time: {vmd_time:.4f} seconds")
print(f"  - Breathing rate: {breathing_rate_vmd_hz:.3f} Hz ({breathing_rate_vmd_bpm:.2f} BPM)")
print(f"  - Selected mode: {dominant_mode_idx+1}")
print(f"  - Mode center frequency: {omega_vmd[dominant_mode_idx]:.3f} Hz")
# =============================================
# 4. WAVELET TRANSFORM (Traditional DWT + FFT)
# =============================================
print("\n" + "="*50)
print("4. WAVELET TRANSFORM (Traditional DWT + FFT)")
print("="*50)

def analyze_wavelet_breathing(signal, fs, wavelet='db4', max_level=4):
    """Traditional DWT-based breathing component extraction with FFT frequency estimation"""
    
    # Determine optimal decomposition level
    optimal_level = min(max_level, int(np.log2(len(signal))))
    print(f"Wavelet: Using {optimal_level} decomposition levels with {wavelet} wavelet")
    
    # Perform multilevel decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=optimal_level, mode='periodization')
    
    # Get coefficient names
    coeff_names = [f'A{optimal_level}'] + [f'D{i}' for i in range(optimal_level, 0, -1)]
    
    # Analyze frequency content of each component to find best breathing component
    best_breathing_component = None
    best_breathing_rate = 0
    best_component_name = ""
    max_resp_power = 0
    
    print("\nAnalyzing wavelet components:")
    print("Component | Freq Range (Hz) | Resp Power | Dominant Freq (Hz) | BPM")
    print("-" * 65)
    
    for i, (coeff, name) in enumerate(zip(coeffs, coeff_names)):
        if coeff is None or len(coeff) < 10:
            continue
            
        # Calculate effective sampling rate for this level
        if name.startswith('A'):  # Approximation
            level = int(name[1:])
            effective_fs = fs / (2 ** level)
        else:  # Detail
            level = int(name[1:])
            effective_fs = fs / (2 ** (level - 1))
        
        # Reconstruct this component only
        test_coeffs = [np.zeros_like(c) if j != i else c for j, c in enumerate(coeffs)]
        reconstructed = pywt.waverec(test_coeffs, wavelet, mode='periodization')
        
        # Trim to original length
        if len(reconstructed) > len(signal):
            reconstructed = reconstructed[:len(signal)]
        elif len(reconstructed) < len(signal):
            reconstructed = np.pad(reconstructed, (0, len(signal) - len(reconstructed)))
        
        # FFT analysis of reconstructed component
        fft_vals = np.fft.fft(reconstructed)
        fft_freqs = np.fft.fftfreq(len(reconstructed), d=1/fs)
        positive_mask = fft_freqs >= 0
        pos_freqs = fft_freqs[positive_mask]
        pos_mag = np.abs(fft_vals[positive_mask])
        
        # Respiratory frequency range (0.04-0.6 Hz = 2.4-36 BPM)
        resp_mask = (pos_freqs >= 0.04) & (pos_freqs <= 0.6)
        resp_freqs = pos_freqs[resp_mask]
        resp_mag = pos_mag[resp_mask]
        
        if len(resp_mag) > 0:
            dominant_idx = np.argmax(resp_mag)
            dominant_freq = resp_freqs[dominant_idx]
            resp_power = np.max(resp_mag)
            
            # Frequency range for this component
            freq_range = f"0-{effective_fs/2:.2f}"
            
            print(f"{name:9} | {freq_range:14} | {resp_power:10.2f} | {dominant_freq:17.3f} | {dominant_freq*60:5.2f}")
            
            # Select component with maximum respiratory power
            if resp_power > max_resp_power:
                max_resp_power = resp_power
                best_breathing_component = reconstructed
                best_breathing_rate = dominant_freq
                best_component_name = name
        else:
            freq_range = f"0-{effective_fs/2:.2f}"
            print(f"{name:9} | {freq_range:14} | {'-':10} | {'-':17} | {'-':5}")
    
    # If no good component found, use A4 (lowest frequencies)
    if best_breathing_component is None and f'A{optimal_level}' in dict(zip(coeff_names, coeffs)):
        a4_idx = coeff_names.index(f'A{optimal_level}')
        test_coeffs = [np.zeros_like(c) if j != a4_idx else c for j, c in enumerate(coeffs)]
        best_breathing_component = pywt.waverec(test_coeffs, wavelet, mode='periodization')
        best_component_name = f'A{optimal_level}'
        
        # Calculate frequency for A4
        fft_vals = np.fft.fft(best_breathing_component)
        fft_freqs = np.fft.fftfreq(len(best_breathing_component), d=1/fs)
        resp_mask = (fft_freqs >= 0.04) & (fft_freqs <= 0.6) & (fft_freqs >= 0)
        if np.any(resp_mask):
            resp_freqs = fft_freqs[resp_mask]
            resp_mag = np.abs(fft_vals[resp_mask])
            best_breathing_rate = resp_freqs[np.argmax(resp_mag)]
        else:
            best_breathing_rate = 0.2  # Default fallback
    
    return best_breathing_component, best_component_name, best_breathing_rate

# Apply traditional wavelet analysis
start_time = time.time()
breathing_wavelet, wavelet_level, breathing_rate_wavelet_hz = analyze_wavelet_breathing(
    phi, fs, wavelet='db4', max_level=4
)
wavelet_time = time.time() - start_time

breathing_rate_wavelet_bpm = breathing_rate_wavelet_hz * 60

print(f"\nWavelet Transform Results:")
print(f"  - Computational time: {wavelet_time:.4f} seconds")
print(f"  - Selected component: {wavelet_level}")
print(f"  - Breathing rate: {breathing_rate_wavelet_hz:.3f} Hz ({breathing_rate_wavelet_bpm:.2f} BPM)")

# =============================================
# 5. COMPARISON: SIMPLE WAVELET + FFT APPROACH
# =============================================
print("\n" + "="*50)
print("5. SIMPLE WAVELET + FFT APPROACH")
print("="*50)

# Simple 4-level decomposition
coeffs_simple = pywt.wavedec(phi, 'db4', level=4)
cA4, cD4, cD3, cD2, cD1  = coeffs_simple

# Method 1: Reconstruct using only approximation coefficients
resp_signal_approx = pywt.waverec([cA4, None, None, None, None], 'db4')

# Method 2: Estimate frequency directly from cA4 coefficients
fft_cA4 = np.fft.fft(cA4)
# Calculate effective sampling rate for A4 level
effective_fs_cA4 = fs / (2 ** 4)  # Level 4 approximation
fft_freqs_cA4 = np.fft.fftfreq(len(cA4), d=1/effective_fs_cA4)
positive_freqs_cA4 = fft_freqs_cA4[fft_freqs_cA4 >= 0]
positive_magnitude_cA4 = np.abs(fft_cA4[fft_freqs_cA4 >= 0])

# Respiratory range mask for cA4
resp_mask_cA4 = (positive_freqs_cA4 >= 0.04) & (positive_freqs_cA4 <= 0.6)
resp_freqs_cA4 = positive_freqs_cA4[resp_mask_cA4]
resp_magnitude_cA4 = positive_magnitude_cA4[resp_mask_cA4]

if len(resp_magnitude_cA4) > 0:
    dominant_freq_cA4 = resp_freqs_cA4[np.argmax(resp_magnitude_cA4)]
    print(f"Direct cA4 Analysis: {dominant_freq_cA4:.3f} Hz ({dominant_freq_cA4*60:.2f} BPM)")
else:
    dominant_freq_cA4 = 0
    print("No dominant frequency in cA4 coefficients")

# Method 3: FFT of reconstructed approximation signal
fft_reconstructed = np.fft.fft(resp_signal_approx)
fft_freqs_recon = np.fft.fftfreq(len(resp_signal_approx), d=1/fs)
positive_freqs_recon = fft_freqs_recon[fft_freqs_recon >= 0]
positive_magnitude_recon = np.abs(fft_reconstructed[fft_freqs_recon >= 0])

resp_mask_recon = (positive_freqs_recon >= 0.04) & (positive_freqs_recon <= 0.6)
resp_freqs_recon = positive_freqs_recon[resp_mask_recon]
resp_magnitude_recon = positive_magnitude_recon[resp_mask_recon]

if len(resp_magnitude_recon) > 0:
    dominant_freq_recon = resp_freqs_recon[np.argmax(resp_magnitude_recon)]
    print(f"Reconstructed Signal FFT: {dominant_freq_recon:.3f} Hz ({dominant_freq_recon*60:.2f} BPM)")
else:
    dominant_freq_recon = 0
    print("No dominant frequency in reconstructed signal")

# =============================================
# 6. FINAL COMPARISON
# =============================================
print("\n" + "="*50)
print("6. FINAL COMPARISON")
print("="*50)

print(f"\nAdvanced Wavelet Method:")
print(f"  - Component: {wavelet_level}")
print(f"  - Breathing Rate: {breathing_rate_wavelet_bpm:.2f} BPM")

print(f"\nSimple Wavelet Methods:")
print(f"  - Direct cA4 FFT: {dominant_freq_cA4*60:.2f} BPM")
print(f"  - Reconstructed Signal FFT: {dominant_freq_recon*60:.2f} BPM")

# Choose the most reliable result
rates = [breathing_rate_wavelet_bpm, dominant_freq_cA4*60, dominant_freq_recon*60]
valid_rates = [r for r in rates if r > 5 and r < 40]  # Physiological range

if valid_rates:
    final_rate = np.median(valid_rates)
    print(f"\nFinal Estimated Breathing Rate: {final_rate:.2f} BPM")
else:
    final_rate = breathing_rate_wavelet_bpm
    print(f"\nUsing advanced wavelet result: {final_rate:.2f} BPM")

# =============================================
# SUMMARY OF ALL METHODS
# =============================================
print("\n" + "="*70)
print("SUMMARY OF DECOMPOSITION METHODS")
print("="*70)
print(f"{'Method':<15} {'Time (s)':<12} {'Breathing Rate (Hz)':<20} {'Breathing Rate (BPM)':<20}")
print("-" * 70)
print(f"{'HDMD':<15} {hdmd_time:<12.4f} {breathing_rate_hdmd_hz:<20.3f} {breathing_rate_hdmd_bpm:<20.2f}")
print(f"{'EMD':<15} {emd_time:<12.4f} {breathing_rate_emd_hz:<20.3f} {breathing_rate_emd_bpm:<20.2f}")
print(f"{'VMD':<15} {vmd_time:<12.4f} {breathing_rate_vmd_hz:<20.3f} {breathing_rate_vmd_bpm:<20.2f}")
print(f"{'Wavelet':<15} {wavelet_time:<12.4f} {breathing_rate_wavelet_hz:<20.3f} {breathing_rate_wavelet_bpm:<20.2f}")
print("=" * 70)

print(f"\nOriginal Sampling Frequency: {original_fs:.2f} Hz")
print(f"Resampled Frequency: {fs_desired} Hz")
print(f"Signal Duration: {duration_desired} seconds")

# =============================================
# ENSURE ALL SIGNALS HAVE SAME LENGTH
# =============================================

# Find minimum length among all signals
min_length = min(len(phi), len(breathing_hdmd), len(breathing_emd), 
                len(breathing_vmd), len(breathing_wavelet))

# Trim all signals to same length
phi = phi[:min_length]
breathing_hdmd = breathing_hdmd[:min_length]
breathing_emd = breathing_emd[:min_length]
breathing_vmd = breathing_vmd[:min_length]
breathing_wavelet = breathing_wavelet[:min_length]
t_plot = t[:min_length]

print(f"\nAll signals trimmed to length: {min_length}")

# =============================================
# VISUALIZATION - 5 Separate Plots
# =============================================
print("\n" + "="*50)
print("CREATING 5 SEPARATE PLOTS")
print("="*50)

# Set font properties
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# Define figure size in centimeters (converted to inches)
fig_width = 13.97 / 2.54  # Convert cm to inches
fig_height = (1.5*2.286) / 2.54  # Convert cm to inches

save_path = r"C:\Users\GOPAL\Downloads\signal2"

# 1. Original Signal Plot
fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
ax1.plot(t_plot, phi[:min_length], 'k-', linewidth=1)
ax1.set_xlabel('Time [s]', fontsize=8)
ax1.set_ylabel('Amplitude', fontsize=8)
ax1.set_title('Phase Signal', fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig(f"{save_path}/simulated_signal.svg", format='svg', bbox_inches='tight')
plt.show()

breathing_true = phi[:min_length]  # True breathing signal for comparison
# 2. HDMD Reconstruction Plot
fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
ax2.plot(t_plot, breathing_true, 'k--', linewidth=1, label='Original')
ax2.plot(t_plot, breathing_hdmd, 'r-', linewidth=1, label='HDMD')
ax2.set_xlabel('Time [s]', fontsize=8)
ax2.set_ylabel('Amplitude', fontsize=8)
ax2.set_title(f'HDMD: {breathing_rate_hdmd_bpm:.1f} BPM', fontsize=8)
ax2.legend(fontsize=6)
ax2.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig(f"{save_path}/hdmd_reconstruction.svg", format='svg', bbox_inches='tight')
plt.show()

# 3. EMD Reconstruction Plot
fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
ax3.plot(t_plot, breathing_true, 'k--', linewidth=1, label='Original')
ax3.plot(t_plot, breathing_emd, 'b-', linewidth=1, label='EMD')
ax3.set_xlabel('Time [s]', fontsize=8)
ax3.set_ylabel('Amplitude', fontsize=8)
ax3.set_title(f'EMD: {breathing_rate_emd_bpm:.1f} BPM', fontsize=8)
ax3.legend(fontsize=6)
ax3.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig(f"{save_path}/emd_reconstruction.svg", format='svg', bbox_inches='tight')
plt.show()

# 4. VMD Reconstruction Plot
fig4, ax4 = plt.subplots(figsize=(fig_width, fig_height))
ax4.plot(t_plot, breathing_true, 'k--', linewidth=1, label='Original')
ax4.plot(t_plot, breathing_vmd, 'g-', linewidth=1, label='VMD')
ax4.set_xlabel('Time [s]', fontsize=8)
ax4.set_ylabel('Amplitude', fontsize=8)
ax4.set_title(f'VMD: {breathing_rate_vmd_bpm:.1f} BPM', fontsize=8)
ax4.legend(fontsize=6)
ax4.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig(f"{save_path}/vmd_reconstruction.svg", format='svg', bbox_inches='tight')
plt.show()

# 5. Wavelet Reconstruction Plot
fig5, ax5 = plt.subplots(figsize=(fig_width, fig_height))
ax5.plot(t_plot, breathing_true, 'k--', linewidth=1, label='Original')
ax5.plot(t_plot, breathing_wavelet, 'm-', linewidth=1, label='Wavelet')
ax5.set_xlabel('Time [s]', fontsize=8)
ax5.set_ylabel('Amplitude', fontsize=8)
ax5.set_title(f'WT: {breathing_rate_wavelet_bpm:.1f} BPM', fontsize=8)
ax5.legend(fontsize=6)
ax5.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig(f"{save_path}/wavelet_reconstruction.svg", format='svg', bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("PHASE SIGNAL DECOMPOSITION ANALYSIS COMPLETE!")
print("="*70)
print(f"Processed radar phase signal:")
print(f"  - Original sampling frequency: {original_fs:.2f} Hz")
print(f"  - Resampled frequency: {fs_desired} Hz")
print(f"  - Duration: {duration} seconds")
print(f"  - Total samples: {len(phi)}")
print(f"  - Noise level: 2σ Gaussian noise")
print(f"  - Methods applied: HDMD, EMD, VMD, Wavelet Transform")
print(f"\nBreathing Rate Summary:")
print(f"  - HDMD: {breathing_rate_hdmd_bpm:.2f} BPM ({breathing_rate_hdmd_hz:.3f} Hz)")
print(f"  - EMD: {breathing_rate_emd_bpm:.2f} BPM ({breathing_rate_emd_hz:.3f} Hz)")
print(f"  - VMD: {breathing_rate_vmd_bpm:.2f} BPM ({breathing_rate_vmd_hz:.3f} Hz)")
print(f"  - Wavelet: {breathing_rate_wavelet_bpm:.2f} BPM ({breathing_rate_wavelet_hz:.3f} Hz)")
print(f"\nComputational Time Summary:")
print(f"  - HDMD: {hdmd_time:.4f} seconds")
print(f"  - EMD: {emd_time:.4f} seconds")
print(f"  - VMD: {vmd_time:.4f} seconds")
print(f"  - Wavelet: {wavelet_time:.4f} seconds")
