import math

import matplotlib.pyplot as plt

import torch
import torchaudio

class Cepstrum(torch.nn.Module):
    def __init__(self, waveform: torch.Tensor, sample_rate: int, window_mode: str="hamming", frame_length: int=32, shift_length: int=8, fft_point: int=(2 ** 10), liftering_order: int=2, threshold: float=0.1, device_name: str="cpu") -> None:
        """ Default Constructor

        Args:
            waveform (torch.Tensor[channel, sample]): waveform signal
            sample_rate (int): [Hz]
            window_mode (str): kind of window. Default "hamming"
                -> hamming, hanning, blackman, bartlett, kaiser
            frame_length (int): [ms]. Default 32 [ms]
            shift_length (int): [ms]. Default 8 [ms]
            fft_point (int): [sample]. Default 1024 [sample]
            liftering_order (int): [ms]. Default 2 [ms]
            threshold (float): voicing / unvoicing threshold. Default 0.1
            device_name (str): device name. Default "cpu"
        """
        super().__init__()
        self.waveform = waveform.to(device_name)
        self.sample_rate = sample_rate
        self.window_mode = window_mode
        self.frame_length = int(frame_length * self.sample_rate / 1000)
        self.shift_length = int(self.frame_length / 4) if shift_length == 8 else int(shift_length * self.sample_rate / 1000)
        self.fft_point = fft_point
        self.liftering_order = int(liftering_order * sample_rate / 1000)
        self.threshold = threshold
        self.device_name = device_name

        self.start_calculation_for_properties()
    
    def get_window(self, window_length: int, data_type: torch.dtype=torch.float32) -> torch.Tensor:
        """ get window function (torch.Tensor)

        Args:
            window_length (int): window length
            data_type (torch.dtype): data type. Defaults to torch.float32.

        Returns:
            torch.Tensor: window
        """
        if self.window_mode == "hamming":
            return torch.hamming_window(window_length=window_length, dtype=data_type, device=self.device_name)
        elif self.window_mode == "hanning":
            return torch.hann_window(window_length=window_length, dtype=data_type, device=self.device_name)
        elif self.window_mode == "blackman":
            return torch.blackman_window(window_length=window_length, dtype=data_type, device=self.device_name)
        elif self.window_mode == "bartlett":
            return torch.bartlett_window(window_length=window_length, dtype=data_type, device=self.device_name)
        elif self.window_mode == "kaiser":
            return torch.kaiser_window(window_length=window_length, dtype=data_type, device=self.device_name)

    def calculate_fft_point(self):
        """ calculate proper fft point
        """
        while self.fft_point < (self.frame_length * 2):
            self.fft_point *= 2
    
    def start_calculation_for_properties(self, data_type: torch.dtype=torch.float32):
        """ start calculate for properties
            -> cepstrums [channel, frame, quefrency]
            -> pitches [channel, frame]
            -> spectrum_envelopes [channel, frame, frequency]

        Args:
            data_type (torch.dtype): data type. Defaults to torch.float32.
        """
        self.calculate_fft_point()
        total_frame_number = math.ceil(((self.waveform.shape[1] - self.frame_length) / self.shift_length) + 1)
        self.cepstrums = torch.zeros(self.waveform.shape[0], total_frame_number, self.fft_point, dtype=data_type, device=self.device_name)
        self.pitches = torch.zeros(self.waveform.shape[0], total_frame_number, dtype=data_type, device=self.device_name)
        self.spectrum_envelopes = torch.zeros(self.waveform.shape[0], total_frame_number, int(self.fft_point / 2) + 1, dtype=data_type, device=self.device_name)
        waveform = torch.zeros(self.waveform.shape[0], (total_frame_number - 1) * self.shift_length + self.frame_length)
        waveform[:][:self.waveform.shape[1] - 1] = self.waveform
        waveform = waveform.to(self.device_name)

        for channel_index in range(waveform.shape[0]):
            for frame_index in range(total_frame_number):
                signal = waveform[channel_index][(frame_index * self.shift_length) : (frame_index * self.shift_length + self.frame_length)].to(self.device_name)
                signal = signal * self.get_window(window_length=self.frame_length, data_type=data_type).to(self.device_name)
                self.cepstrums[channel_index][frame_index] = torch.fft.ifft(torch.log(torch.abs(torch.fft.fft(signal, n=self.fft_point))), n=self.fft_point).real.to(self.device_name)
                pitch_index = torch.argmax(self.cepstrums[channel_index][frame_index][self.liftering_order: ]) + self.liftering_order
                if self.threshold <= self.cepstrums[channel_index][frame_index][pitch_index]:
                    self.pitches[channel_index][frame_index] = 1 / (self.cepstrums[channel_index][frame_index][pitch_index] * 1000 / self.sample_rate)
                cepstrums = self.cepstrums[channel_index][frame_index]
                cepstrums[channel_index][frame_index][self.liftering_order + 1: -self.liftering_order] = 0
                self.spectrum_envelopes[channel_index][frame_index] = torch.exp(torch.fft.fft(cepstrums[channel_index][frame_index], n=self.fft_point)[: int(self.fft_point / 2) + 1].real.to(self.device_name))

    def __str__(self) -> str:
        string = " " + ("-" * 40) + " Cepstrum " + ("-" * 40) + "\n"
        string += f"waveform shape : {self.waveform.shape} [channel, sample]\n"
        string += f"sample_rate : {self.sample_rate} [Hz]\n"
        string += f"window_mode : {self.window_mode}\n"
        string += f"frame_length : {self.frame_length} [sample]\n"
        string += f"shift_length : {self.shift_length} [sample]\n"
        string += f"fft_point : {self.fft_point} [sample]\n"
        string += f"liftering_order : {self.liftering_order} [sample]\n"
        string += f"threshold : {self.threshold}\n"
        string += f"device_name : {self.device_name}\n"
        string += f"cepstrums shape : {self.cepstrums.shape} [channel, frame, quefrency]\n"
        string += f"pitches shape : {self.pitches.shape} [channel, frame]\n"
        string += f"spectrum_envelopes shape : {self.spectrum_envelopes.shape} [channel, frame, frequency]\n"
        string += "-" * 91 + "\n\n"
        return string

    def forward(self) -> None:
        self.start_calculation_for_properties()
        return self.cepstrums, self.pitches, self.spectrum_envelopes

def test_cepstrum(waveform: torch.Tensor, sample_rate: int, window_mode: str, frame_length: int, shift_length: int, fft_point: int, liftering_order: int, threshold: float, device_name: str):
    cepstrum = Cepstrum(
        waveform=waveform,
        sample_rate=sample_rate,
        window_mode=window_mode,
        frame_length=frame_length,
        shift_length=shift_length,
        fft_point=fft_point,
        liftering_order=liftering_order,
        threshold=threshold,
        device_name=device_name
    )
    print(cepstrum)

def main():
    audio_file_path = "./Sources/Audio/sample.wav"
    frame_length = 32   # [ms]
    shift_length = int(frame_length / 4) # [ms]
    window_mode = "hamming"
    device_name = "cpu"
    fft_point = 2 ** 14
    liftering_order = 4
    threshold = 0.075
    waveform, sample_rate = torchaudio.load(audio_file_path)
    test_cepstrum(
        waveform=waveform,
        sample_rate=sample_rate,
        window_mode=window_mode,
        frame_length=frame_length,
        shift_length=shift_length,
        fft_point=fft_point,
        liftering_order=liftering_order,
        threshold=threshold,
        device_name=device_name
    )

if __name__ == "__main__":
    main()