import sys
sys.path.append('waveglow/')
import csv
import os
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

# Parse command-line arguments
n = len(sys.argv)
input = "input.csv"
output = "data/"

if n != 1 and n != 3:
    print("Invalid command-line arguments, using default locations for input and output")
elif n == 3:
    input = sys.argv[1]
    output = output + sys.argv[2] + "/"

# Create output directory if not exists
if not os.path.isdir(output):
    os.mkdir(output)

# Load Tacotron2 and Waveglow pretrained models from NVIDIA
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

# Simple function for writing generating audio from input and writing to file
def write_to_file(input, path, count):
    # Prepare input and run inference
    sequence = np.array(text_to_sequence(input + ".", ['english_cleaners']))[None, :] # Appending period results in more consistent audio
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.75)

    # Denoise/normalize audio and write to .wav file
    audio = denoiser(audio, strength=0.01)[:, 0]
    audio = audio[0].data.cpu().numpy()
    m = np.max(np.abs(audio))
    audio = (audio/m).astype(np.float32)
    print("Writing \"%s\"" % input)
    if count == 0:
        write(path + input + ".wav", hparams.sampling_rate, audio)
    else:
        write(path + input + "_" + str(count) + ".wav", hparams.sampling_rate, audio)

# Loop over csv and generate data
with open(input) as  csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        input = row[0]
        count = int(row[1])
        if count == 1:
            write_to_file(input, output, 0)
        else:
            path = output + input + "/"
            if not os.path.isdir(path):
                os.mkdir(path)
            for i in range(1, count+1):
                write_to_file(input, path, i)
    print("Data generation complete")
