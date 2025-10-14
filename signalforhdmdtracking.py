import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, stft
from scipy.linalg import hankel, svd, eig

# Set font properties
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# Define figure size in centimeters (converted to inches)
fig_width = 13.97 / 2.54  # Convert cm to inches
fig_height = (1.5*2.286) / 2.54  # Convert cm to inches

# Parameters
fs = 120            # Sampling rate in Hz
duration = 120    # Signal duration in seconds (now 120s for 3 intervals)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector

# Initialize empty signal
phi = np.zeros_like(t)

# Component 1: Main breathing component (0-40s)
interval1 = t < 40
amp1 = 5.0 + 0.2 * np.sin(2 * np.pi * 0.02 * t[interval1])  # Slow amplitude modulation (0.02 Hz)
freq1 = 0.25  # 15 BPM
comp1 = amp1 * np.cos(2 * np.pi * freq1 * t[interval1])

# Component 2: Heart rate component (constant throughout)
freq2 = 1.2  # 72 BPM
comp2 = 1 * np.cos(2 * np.pi * freq2 * t)

# Component 3: High-frequency noise (constant throughout)
freq3 = 5.0  # 5 Hz
comp3 = 0.005 * np.cos(2 * np.pi * freq3 * t)

# Interval 1 (0-40s)
phi[interval1] = comp1 + comp2[interval1] + comp3[interval1]

# Interval 2 (40-80s): Different breathing frequency
interval2 = (t >= 40) & (t < 80)
amp2 = 5.0 + 0.3 * np.sin(2 * np.pi * 0.015 * t[interval2])  # Different modulation
freq4 = 17/60  # ≈0.283 Hz (17 BPM)
comp4 = amp2 * np.cos(2 * np.pi * freq4 * (t[interval2]-40))  # Note: t-40 for phase continuity
phi[interval2] = comp4 + comp2[interval2] + comp3[interval2]

# Interval 3 (80-120s): Different breathing frequency
interval3 = t >= 80
amp3 = 5.0 + 0.15 * np.sin(2 * np.pi * 0.025 * t[interval3])  # Different modulation
freq5 = 12/60  # 0.2 Hz (12 BPM)
comp5 = amp3 * np.cos(2 * np.pi * freq5 * (t[interval3]-80))  # Note: t-80 for phase continuity
phi[interval3] = comp5 + comp2[interval3] + comp3[interval3]

# Add Gaussian noise
np.random.seed(42)  # For reproducibility
noise = 0.5*np.random.randn(len(t))
phi += noise

# Create true breathing component for comparison
breathing_true = np.zeros_like(t)
breathing_true[interval1] = comp1
breathing_true[interval2] = comp4
breathing_true[interval3] = comp5

# ===========================
# STFT SPECTROGRAM ANALYSIS
# ===========================

print("Computing STFT spectrogram...")

# Compute STFT
nperseg = min(512, len(phi)//8)  # Window size for STFT
noverlap = nperseg // 2  # 50% overlap
frequencies, times, Zxx = stft(phi, fs=fs, nperseg=nperseg, noverlap=noverlap)

# Convert power to dB scale
Zxx_magnitude = np.abs(Zxx)
Zxx_db = 20 * np.log10(Zxx_magnitude + 1e-10)  # Add small value to avoid log(0)

# ===========================
# DMD-t TRACKING IMPLEMENTATION
# ===========================

def hankel_dmd(X, Y, rank=None):
    """
    Perform DMD on Hankel matrices X and Y
    """
    try:
        # SVD of X
        U, s, Vh = svd(X, full_matrices=False)
        
        # Determine rank
        if rank is None:
            # Automatic rank selection based on singular value threshold
            cumsum = np.cumsum(s) / np.sum(s)
            rank = np.where(cumsum >= 0.99)[0][0] + 1
            rank = min(rank, min(X.shape) - 1)
        
        # Truncate SVD
        r = min(rank, len(s))
        U_r = U[:, :r]
        s_r = s[:r]
        Vh_r = Vh[:r, :]
        
        # Compute A_tilde
        A_tilde = U_r.T @ Y @ Vh_r.T @ np.diag(1/s_r)
        
        # Eigendecomposition of A_tilde
        eigenvalues, W = eig(A_tilde)
        
        # Compute DMD modes
        modes = Y @ Vh_r.T @ np.diag(1/s_r) @ W
        
        # Compute frequencies from eigenvalues
        dt = 1/fs  # time step
        frequencies = np.log(eigenvalues) / (2j * np.pi * dt)
        frequencies = np.real(frequencies)  # Take real part for frequencies
        
        # Compute amplitudes (initial conditions)
        try:
            amplitudes = np.linalg.pinv(modes) @ X[:, 0]
        except:
            amplitudes = np.ones(len(eigenvalues))
        
        return modes, eigenvalues, frequencies, amplitudes
    except:
        return None, None, None, None

def dmdt_tracking(signal, fs, window_size=5.0, step_size=1.0, rank=None):
    """
    Perform DMD-t tracking on a signal with sliding windows
    """
    n_samples = len(signal)
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    
    # Calculate number of windows
    n_windows = int((n_samples - window_samples) / step_samples) + 1
    
    # Initialize results
    time_points = []
    frequencies = []
    amplitudes = []
    
    for i in range(n_windows):
        # Extract current window
        start = i * step_samples
        end = start + window_samples
        if end > n_samples:
            break
        window_signal = signal[start:end]
        
        # Build Hankel matrix
        m = min(150, window_samples//2)  # rows in Hankel matrix
        n = window_samples - m + 1       # columns
        
        if n <= 0:
            continue
            
        H = hankel(window_signal[:n], window_signal[n-1:])
        
        # Split into X and Y matrices
        X = H[:, :-1]
        Y = H[:, 1:]
        
        if X.shape[1] == 0:
            continue
        
        # Perform DMD
        try:
            modes, evals, freqs, _ = hankel_dmd(X, Y, rank=rank)
            
            if freqs is not None:
                # Store results
                time_points.append((start + end) / (2 * fs))  # center time
                frequencies.append(freqs)
                amplitudes.append(np.abs(evals))
        except:
            continue
    
    return np.array(time_points), frequencies, amplitudes

# Perform DMD-t tracking
print("\nPerforming DMD-t tracking...")
window_size = 5.0  # 5-second windows
step_size = 1.0    # 1-second steps
time_points, all_frequencies, all_amplitudes = dmdt_tracking(phi, fs, window_size, step_size)

# Process results to find dominant frequencies
dominant_freqs = []
dominant_amps = []

for freqs, amps in zip(all_frequencies, all_amplitudes):
    if freqs is None:
        dominant_freqs.append(np.nan)
        dominant_amps.append(np.nan)
        continue
        
    # Focus on respiratory band (0.1-0.8 Hz)
    resp_mask = (np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.8)
    resp_freqs = freqs[resp_mask]
    resp_amps = amps[resp_mask]
    
    if len(resp_freqs) > 0:
        # Find dominant frequency in respiratory band
        dominant_idx = np.argmax(resp_amps)
        dominant_freqs.append(resp_freqs[dominant_idx])
        dominant_amps.append(resp_amps[dominant_idx])
    else:
        dominant_freqs.append(np.nan)
        dominant_amps.append(np.nan)

dominant_bpm = np.abs(dominant_freqs) * 60

# ===========================
# CREATE THE PLOTS
# ===========================

# First plot: DMD-t and Spectrogram side by side
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# Plot 1: DMD-t breathing rate tracking
if len(time_points) > 0 and len(dominant_bpm) > 0:
    valid_mask = ~np.isnan(dominant_bpm)
    if np.any(valid_mask):
        ax1.plot(time_points[valid_mask], dominant_bpm[valid_mask], 'b-', linewidth=1.5, 
                marker='o', markersize=3, label='DMD-t Estimated Rate')
        ax1.set_ylabel('Breathing Rate [BPM]', fontsize=8)
        ax1.set_title('DMD-t Tracking', fontsize=8)
        ax1.grid(False)
        
        # Set reasonable y-limits based on data
        valid_bpm = dominant_bpm[valid_mask]
        if len(valid_bpm) > 0:
            y_min = max(5, np.min(valid_bpm) - 5)
            y_max = min(60, np.max(valid_bpm) + 5)
            ax1.set_ylim([y_min, y_max])
        else:
            ax1.set_ylim([10, 40])
    else:
        ax1.text(0.5, 0.5, 'No valid DMD-t estimates', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=10)
        ax1.set_title('DMD-t Tracking - No Valid Results', fontsize=10)
else:
    ax1.text(0.5, 0.5, 'DMD-t analysis failed', ha='center', va='center', 
             transform=ax1.transAxes, fontsize=10)
    ax1.set_title('DMD-t Tracking - Analysis Failed', fontsize=8)
ax1.set_xlabel('Time [s]', fontsize=8)

# Plot 2: STFT Spectrogram (focused on breathing band in BPM)
breathing_mask = (frequencies >= 0.1) & (frequencies <= 0.8)
im = ax2.pcolormesh(times, frequencies[breathing_mask] * 60, Zxx_db[breathing_mask, :], 
                 shading='gouraud', cmap='plasma')
cbar = plt.colorbar(im, ax=ax2, label='Magnitude [dB]', shrink=0.8)

cbar.set_label('Magnitude [dB]', size=8)
ax2.set_ylabel('Breathing Rate [BPM]', fontsize=8)
ax2.set_xlabel('Time [s]', fontsize=8)
ax2.set_title('STFT', fontsize=8)
ax2.set_ylim([10, 40])  # 6-48 BPM range
ax2.grid(False, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
save_path1 = r"C:\Users\GOPAL\Downloads\dmd_spectrogram.svg"
plt.savefig(save_path1, format='svg', dpi=300)
plt.show()

# Second plot: Just the signal in black color
fig2, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
ax.plot(t, phi, color='black', linewidth=0.5)
ax.set_xlabel('Time [s]', fontsize=8)
ax.set_ylabel('Amplitude', fontsize=8)
ax.set_title('Synthetic Signal', fontsize=8)
ax.axvline(x=40, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=80, color='red', linestyle='--', alpha=0.5)
ax.grid(False)

# Adjust layout and save
plt.tight_layout()
save_path2 = r"C:\Users\GOPAL\Downloads\signal.svg"
plt.savefig(save_path2, format='svg', dpi=300)
plt.show()

print(f"First plot saved to {save_path1}")
print(f"Second plot saved to {save_path2}")