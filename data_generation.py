import sys
sys.path.append('waveglow/')
import csv
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from scipy.io.wavfile import write

#  Load Tacotron2 and Waveglow pretrained models from NVIDIA
hparams = create_hparams()
hparams.sampling_rate = 22050
hparams.max_decoder_steps = 2500

checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

# Setup for data generation
input = "input.csv"
output = "data/"

# Simple function for writing generating audio from input and writing to file
def write_to_file(input):
    # Prepare input and run inference
    sequence = np.array(text_to_sequence(input + ".", ['english_cleaners']))[None, :] # Appending period results in more consistent audio
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.5)

    # Denoise/normalize audio and write to .wav file
    # audio = denoiser(audio, strength=0.05)[:, 0]
    audio = audio[0].data.cpu().numpy()
    m = np.max(np.abs(audio))
    audio = (audio/m).astype(np.float32)
    print("Writing \"%s\"" % input)
    write(output + input + ".wav", hparams.sampling_rate, audio)

# Loop over csv and generate data
with open("input.csv") as  csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        write_to_file(row[0])
    print("Data generation complete")
