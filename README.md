HDMD README

This repository contains the implementation of Hankel Dynamic Mode Decomposition (Hankel-DMD) for radar-based respiratory sensing and tracking using the Acconeer A121 radar sensor.

Installation
Prerequisites

Python 3.8 or higher
Acconeer A121 Radar Module (XE125)
USB cable for radar connection

Dependencies
Install required packages:
bashpip install numpy>=1.20 matplotlib>=3.4 scipy>=1.7 pywt>=1.1 h5py>=3.6
pip install --upgrade acconeer-exptool[app]>=4.9.0

Note: Depending on your environment, replace pip with pip3 or use python -m pip.

Driver Installation
If no COM port is recognized when plugging in the module, you may need to install a driver. See Acconeer module-specific information for details.

Hardware Setup
1. Initial Setup
Run the setup wizard:
bashpython -m acconeer.exptool.setup
2. Launch Acconeer Exploration Tool
bashpython -m acconeer.exptool.app
3. Configure Connection

Select A121 as the sensor type
Set port type to Serial
Select port XE125 (typically /dev/ttyUSB0 on Linux)
Click Connect

4. Flash the Radar (First-Time Setup Only)
If this is your first time using the radar:

Navigate to the Flash tab
Select Get latest binary for A121
Put the radar module in bootloader mode:

Press and hold the DFU button
Press the RESET button (while still holding DFU)
Release the RESET button
Release the DFU button


Follow on-screen instructions to complete flashing
Press and release the RESET button after flashing completes

5. Configure Radar Parameters
Navigate to the Stream tab and select Sparse IQ service. Configure the following parameters:
Metadata

Frame data length: 40
Max sweep rate: 2575 Hz

Processor Parameters

Amplitude method: Coherent

Sensor Parameters

Sweeps per frame: 1

Subsweep Parameters

Start point: 80
Number of points: 40
Step length: 8
HWAAS: 8
Profile: 3
Receiver gain: 16
PRF: 15.6 MHz
Enable transmitter: ✓

