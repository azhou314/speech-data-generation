# Speech Data Generation

A quick script to generate audio speech data using NVIDIA's Tacotron 2 and
WaveGlow models.


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone https://github.com/azhou314/speech-data-generation.git`
2. CD into this repo: `cd speech-data-generation`
3. Initialize WaveGlow submodule: `git submodule init; git submodule update`
4. Download pretrained [Tacotron](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view) and [WaveGlow](https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view) models from NVIDIA and place into the repo

## Data generation
1. Create a `.csv` file of the desired speech data. The file should have two columns. The first column should be of the words/phrases to be generated (without punctuation), and the second column should contain the number of times to sample each word or phrase.
    - Words or phrases can be specified in conventional English orthography, or in ARPABET
    - To specify words/phrases in ARPABET, surround with curly braces and use 2-letter codes:
        - The list of valid 2-letter codes is found below, where numbers are appended to vowels to signify stress:
            ```
            valid_symbols = ['AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
                             'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
                             'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
                             'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
                             'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
                             'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
            ```
        - The corresponding IPA symbols for each code can be found [here](https://en.wikipedia.org/wiki/ARPABET)
    - An example of a valid `.csv` is found in `input.csv`
2. Run `python data_generation.py CSV_FILE_NAME OUTPUT_FOLDER_LOCATION`
    - By default, the script will look at `input.csv` and output data directly into `/data`
        - Otherwise, the generated data will be found in `/data/OUTPUT_FOLDER_LOCATION/`
    - Pre-generated data using `input.csv` can be found in `/data`

## Acknowledgements
This code exclusively uses code from NVIDIA's Tacotron 2 [implemention](https://github.com/NVIDIA/tacotron2) 
