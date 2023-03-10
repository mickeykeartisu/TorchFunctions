# TorchFunctions

## Cepstrum

---

- Default Constructor Arguments :
  - waveform (torch.Torch) : audio signal shape [channel, sample]
  - sample_rate (int) : sample_rate [Hz]
  - window_mode (str) : kind of window function. Default "hamming"
    1. hamming
    2. hanning
    3. blackman
    4. bartlett
    5. kaiser
  - frame_length (int): [ms]. Default 32 [ms]
  - shift_length (int): [ms]. Default 8 [ms]
  - fft_point (int): [sample]. Default 1024 [sample]
  - liftering_order (int): [ms]. Default 2 [ms]
  - threshold (float): voicing / unvoicing threshold. Default 0.1
  - device_name (str): device name. Default "cpu"

- Properties :
  - waveform (torch.Torch) : audio signal shape [channel, sample]
  - sample_rate (int) : sample_rate [Hz]
  - window_mode (str) : kind of window function. Default "hamming"
    1. hamming
    2. hanning
    3. blackman
    4. bartlett
    5. kaiser
  - frame_length (int): [sample].
  - shift_length (int): [sample].
  - fft_point (int): [sample].
  - liftering_order (int): [sample].
  - threshold (float): voicing / unvoicing threshold
  - device_name (str): device name. Default "cpu"
  - cepstrums (torch.Tensor) : cepstrums [channel, frame, quefrency]
  - pitches (torch.Tensor) : pitches [channel, frame]
  - spectrum_envelopes (torch.Tensor) : spectrum envelopes [channel, frame, frequency]

---
