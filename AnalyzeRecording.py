#from recorder_GUI import analyzer
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../computeDescriptorsDir'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../violinDemoRT'))
import  utilFunctions as UF
import harmonicModel as HM
from computeDescriptors import energyInBands, computeHarmonicEnvelope
# Librosa for audio
import librosa
import struct
from scipy.io.wavfile import write, read

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}



class Analyzer_MFCCs:
    def analize(self, resources_dir="", filename="", windowSize = 1024, hopSize = 128, fftPad = 1, normalize=0):
        #n_classes = 5

        fftSize = windowSize * fftPad

        inputFile = resources_dir + '/' + filename + '.wav'
        (fs, x) = wavread(inputFile)

        # Mel-scaled power (energy-squared) spectrogram
        Sxx = librosa.feature.melspectrogram(x, sr=fs, n_mels=128, n_fft=fftSize,
                                             hop_length=hopSize)
        # n_fft=2048, hop_length=512,power=2.0,
        # Convert to log scale (dB). We'll use the peak power as reference.
        SxxdB = librosa.logamplitude(Sxx, ref_power=np.max)
        if normalize:
            auxavg = np.average(SxxdB, axis=0)
            SxxdBNorm = SxxdB / auxavg
        else:
            SxxdBNorm = SxxdB.T

        #output
        output_file = resources_dir + '/' + filename + '_MFCCs.txt'
        np.savetxt(output_file, SxxdBNorm, fmt='%.5f', delimiter=' ', newline='\n', header='', footer='',
                   comments='%MFCCs (dB), normalized to mean energy of the frame')
        print("Saved output file: ", output_file)
        return

class Analyzer_40_Band:
    #def __init__(self):

    def analyze(self, resources_dir="", filename=""):
        N = 2048  # fftSize
        hop = 256 #128:used before, shouldnt be 183,xx?(240Hz)  # hop Size
        minFFTVal = -120
        bandCentersHz = np.array(
            [103, 171, 245, 326, 413, 508, 611, 722, 843, 975, 1117, 1272, 1439, 1621, 1819, 2033, 2266, 2518, 2792,
             3089,
             3412, 3761, 4141, 4553, 5000, 5485, 6011, 6582, 7202, 7874, 8604, 9396, 10255, 11187, 12198, 13296, 14487,
             15779, 17181, 18703])

        inputFile = resources_dir + '/' + filename + '.wav'
        (fs, x) = UF.wavread(inputFile)
        NyqFreq = fs / 2
        hfreq, hmag = HM.harmonicAnalisys(x, fs, H=hop, t=minFFTVal)
        nFrames =  hfreq.shape[0]
        freqs = np.array(range(0, int(N / 2 + 1))) * fs / N
        freqs[-1] = NyqFreq - 1

        harmonicEnvelope = np.zeros(shape=(nFrames, N / 2 + 1))
        energyBand = np.zeros(shape=(nFrames, len(bandCentersHz)))
        for iFrame in range(0, nFrames):
            # # compute harmonic envelope as spline
            harmonicEnvelope[iFrame, :] = computeHarmonicEnvelope(hfreq[iFrame, :], hmag[iFrame, :], NyqFreq,
                                                                  minFFTVal,
                                                                  N, freqs)
            energyBand[iFrame, :] = energyInBands(harmonicEnvelope[iFrame, :], bandCentersHz, fs, minFFTVal)
            #if iFrame % 100 == 0:
            #    print("Filename:", filename, ", frame:", iFrame, "/", nFrames)

        outputFile = resources_dir + '/' + filename + '.EnergyBankFilter_hop' + str(hop) + '.txt'
        #print('output file:', outputFile)
        np.savetxt(outputFile, energyBand, fmt='%.5f', delimiter=' ', newline='\n', header='', footer='',
                   comments='%40 energy bank filter (dB)')  # [source]
        print("Saved output file: ", outputFile)

        return


def wavread(filename):
    """
    Read a sound file and convert it to a normalized floating point array
    filename: name of file to read
    returns fs: sampling rate of file, x: floating point array
    """

    if (os.path.isfile(filename) == False):  # raise error if wrong input file
        raise ValueError("Input file is wrong")

    [fs, x] = read(filename)

    if (len(x.shape) != 1):  # raise error if more than one channel
        raise ValueError("Audio file should be mono")

    if (fs != 44100):  # raise error if more than one channel
        raise ValueError("Sampling rate of input sound should be 44100")

        # scale down and convert audio into floating point number in range of -1 to 1
    x = np.float32(x) / norm_fact[x.dtype.name]
    return fs, x

if __name__ == "__main__":
    doBands = 1
    if doBands:
        print('Bands')
        if False:
            #resources_dir = './resources/MacMic_MyViolin/'
            resources_dir = './resources/Scarlett_MyViolin/'
            #filenames = ['EStringScaleVibrato', 'AStringScaleVibrato', 'DStringScaleVibrato', 'GStringScaleVibrato']
            #filenames = ['EStringGliss', 'AStringGliss', 'DStringGliss', 'GStringGliss']
            filenames = ['BackNoiseMaia', 'EStringGliss', 'AStringGliss', 'DStringGliss', 'GStringGliss',
                         'EStringScaleDetache', 'AStringScaleDetache', 'DStringScaleDetache', 'GStringScaleDetache',
                         'EStringScaleVibrato', 'AStringScaleVibrato', 'DStringScaleVibrato', 'GStringScaleVibrato'
                         ]
            filenames = ['EString_Long', 'AString_Long', 'DString_Long', 'GString_Long']
            analyzer = Analyzer_40_Band()
            #print("Analizing file #", iString + 1, "/", len(filenames), ":", resources_dir, filenames[iString], ".wav")
            analyzer.analyze(resources_dir, filenames)
        else:
            score_list = 'recording_script.scoreList'
            base_dir = '/Users/alfonso/recordings/recordingsYamaha/phrases/'
            analyzer = Analyzer_40_Band()
            filenames = []
            with open(base_dir + score_list) as f:
                for line in f:
                    filenames.append(line)
            filenames = [x.strip() for x in filenames]
            for i_file in range(0, len(filenames)):
                prefix_dir = 'tools_phrases_'
                analyzer.analyze(base_dir + prefix_dir + filenames[i_file], filenames[i_file] + '-16bit')

    else:
        print('MFCCs')
        score_list = 'recording_script.scoreList'
        base_dir = '/Users/alfonso/recordings/recordingsYamaha/phrases/'
        analyzer = Analyzer_MFCCs()
        filenames = []
        with open(base_dir + score_list) as f:
            for line in f:
                filenames.append(line) #filenames = f.readlines()
        filenames = [x.strip() for x in filenames]
        for i_file in range(0, len(filenames)):
            prefix_dir = 'tools_phrases_'
            analyzer.analize(base_dir + prefix_dir + filenames[i_file], filenames[i_file] + '-16bit')

    print("Analysis DONE!")



