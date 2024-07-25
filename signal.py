import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt

class TimeSignal:
    def __init__(self, time, signal, sampling_rate):
        self.time = time
        self.signal = signal
        self.sampling_rate = sampling_rate

    def __repr__(self):
        return f"TimeSignal(sampling_rate={self.sampling_rate}, duration={self.time[-1] - self.time[0]}s)"

    def plot_signal(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.time, self.signal)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Time Signal')
        plt.grid(True)
        plt.show()

    def fft(self):
        N = len(self.signal)
        T = 1.0 / self.sampling_rate
        yf = fft(self.signal)
        xf = fftfreq(N, T)[:N//2]
        return xf, 2.0/N * np.abs(yf[0:N//2])

    def plot_fft(self):
        xf, yf = self.fft()
        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Frequency Spectrum')
        plt.grid(True)
        plt.show()

    def filter_signal(self, lowcut, highcut, order=5):
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, self.signal)
        return TimeSignal(self.time, filtered_signal, self.sampling_rate)

    def basic_statistics(self):
        return {
            'mean': np.mean(self.signal),
            'std_dev': np.std(self.signal),
            'max': np.max(self.signal),
            'min': np.min(self.signal)
        }

    def save_to_file(self, filename):
        np.savetxt(filename, np.column_stack((self.time, self.signal)), delimiter=',', header='time,signal', comments='')

    @classmethod
    def load_from_file(cls, filename, sampling_rate):
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        time = data[:, 0]
        signal = data[:, 1]
        return cls(time, signal, sampling_rate)
