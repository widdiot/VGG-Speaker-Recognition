from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import sys
import kaldiio
sys.path.append('../tool')
import toolkits
#import utils as ut
import glob
import pdb
import os
import save_segs
from kaldi.feat.mfcc import MfccOptions, Mfcc
from kaldi.feat.window import FrameExtractionOptions
from kaldi.feat.mel import MelBanksOptions
from kaldi.matrix import Vector, SubVector, DoubleVector, Matrix
from kaldi.util.table import MatrixWriter, SequentialMatrixReader, RandomAccessPosteriorReader, SequentialWaveReader, \
    DoubleVectorWriter, SequentialVectorReader
from kaldi.ivector import compute_vad_energy, VadEnergyOptions
from kaldi.feat.functions import sliding_window_cmn, SlidingWindowCmnOptions
from kaldi.ivector import LogisticRegressionConfig, LogisticRegression
from kaldi.feat.functions import sliding_window_cmn, SlidingWindowCmnOptions
from kaldi.matrix import Vector, SubVector, DoubleVector, Matrix
from kaldi.util.io import xopen
from kaldi.matrix.common import MatrixTransposeType
import time
# ===========================================
#        Parse the argument
# ===========================================
import argparse

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='../model_tandem/gvlad_softmax/2021-02-06_resnet34s_bs16_adam_lr0.001_vlad10_ghost2_bdim512_ohemlevel0/', type=str)
parser.add_argument('--task', default='lre', choices=['lre', 'sre'], type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--labels', default='../model_tandem/gvlad_softmax/2021-02-06_resnet34s_bs16_adam_lr0.001_vlad10_ghost2_bdim512_ohemlevel0/label2idx', type=str)
parser.add_argument('--data_path', default='', type=str)
parser.add_argument('--file_list', default='/home/gnani/LID/TEST/test_files', type=str)
# set up network configuration.
parser.add_argument('--cmvn', default='/home/gnani/LID/train/cmvn.ark', type=str)
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax','tuplemax'], type=str)
global args
args = parser.parse_args()

# ==================================
#       Get Model
# ==================================
# construct the data generator.
params = {'dim': (40, None, 1),
          'nfft': 512,
          'spec_len': 500,
          'win_length': 400,
          'hop_length': 160,
          'n_classes': 10,
          'sampling_rate': 8000,
          'normalize': False,
          }

toolkits.initialize_GPU(args)
import model
network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                            num_class=params['n_classes'],
                                            mode='eval', args=args)
import kaldiio
if args.cmvn:
    cmvn_stats = kaldiio.load_mat(args.cmvn)
    mean_stats = cmvn_stats[0,:-1]
    count = cmvn_stats[0,-1]
    offset = np.expand_dims(mean_stats,0)/count
    print("offset",offset)
    CMVN = offset
#print(network_eval.summary())
# ==> load pre-trained model ???
if args.resume:
    # ==> get real_model from arguments input,
    # load the model if the imag_model == real_model.
    if os.path.isfile(args.resume):
        network_eval.load_weights(os.path.join(args.resume), by_name=True)
#        result_path = set_result_path(args)
        print('==> successfully loading model {}.'.format(args.resume))
    else:
        raise IOError("==> no checkpoint found at '{}'".format(args.resume))
else:
    raise IOError('==> please type in the model to load')

print('==> start testing.')

############### get labels file #################
with open(args.labels) as f:
    lines = f.read().splitlines()
label2idx = {}
for l in lines:
    label2idx[l.split()[0]] = int(l.split()[1])
i2l = {}
for key, val in label2idx.items():
    i2l[val] = key

############## Mfcc opts #################
fopts = FrameExtractionOptions()
fopts.samp_freq = 8000
fopts.snip_edges = False

hires_mb_opts = MelBanksOptions()
hires_mb_opts.low_freq = 40
hires_mb_opts.high_freq = -200
hires_mb_opts.num_bins = 40
hires_mfcc_opts = MfccOptions()
hires_mfcc_opts.frame_opts = fopts
hires_mfcc_opts.num_ceps = 40
hires_mfcc_opts.mel_opts = hires_mb_opts
hires_mfcc_opts.use_energy = False

############## Sliding Window opts #################
sliding_windows_opts = SlidingWindowCmnOptions()
sliding_windows_opts.center = True

sliding_windows_opts_low_feats = SlidingWindowCmnOptions()
sliding_windows_opts_low_feats.center = True
sliding_windows_opts_low_feats.min_window = 300
sliding_windows_opts_low_feats.normalize_variance = False
vopts = VadEnergyOptions()


def lid_module(key, audio_file, start, end):
    # ==================================
    #       Get data and process it.
    # ==================================
    wav_spc = "scp:echo " + key + " 'sox -V0 -t wav " + audio_file + " -c 1 -r 8000 -t wav - trim " + str(start) + " " + str(
        float(end) - float(start)) + "|' |"
    hires_mfcc = Mfcc(hires_mfcc_opts)
    wav = SequentialWaveReader(wav_spc).value()
    hi_feat = hires_mfcc.compute_features(wav.data()[0], wav.samp_freq, 1.0)
    hi_feat = hi_feat.numpy() - CMVN
    X = hi_feat.T
    X = np.expand_dims(np.expand_dims(X, 0), -1)
    #print(X.shape)
    v = network_eval.predict(X)
    #print(v)
    #print(key, "::", i2l[v.argmax()])
    return i2l[v.argmax()]


def set_result_path(args):
    model_path = args.resume
    exp_path = model_path.split(os.sep)
    result_path = os.path.join('../result', exp_path[2], exp_path[3])
    if not os.path.exists(result_path): os.makedirs(result_path)
    return result_path


if __name__ == "__main__":
    if args.data_path:
    	wavs = glob.glob(args.data_path + '*/*.wav')
    	for audio in wavs:
            utt = audio.split('/')[-1][:-4]
       	    Segments, _ = save_segs.build_response(audio)
            for segment in Segments:
                seg_key = utt + "-" + str("{0:.2f}".format(float(segment[0]) / 100)).zfill(7).replace(".", "") + "-" + str("{0:.2f}".format(float(segment[1]) / 100)).zfill(7).replace(".", "")
                start_time = float(segment[0]) / 100
                end_time = float(segment[1]) / 100
                data = {"AudioFile": audio, "startTime": start_time, "endTime": end_time}
                lid_module(seg_key, data["AudioFile"], data["startTime"], data["endTime"])
    elif args.file_list:
        with open(args.file_list) as f:
            file_paths = f.read().splitlines()
        f = open('/'.join(args.file_list.split('/')[:-1])+'/output','w')
        for audio in file_paths:
            if not os.path.exists(audio):
                print("path not found:", audio)
            else:
                utt = audio.split('/')[-1][:-4]
                Segments, _ = save_segs.build_response(audio)
                for segment in Segments:
                    seg_key = utt + "-" + str("{0:.2f}".format(float(segment[0]) / 100)).zfill(7).replace(".", "") + "-" + str("{0:.2f}".format(float(segment[1]) / 100)).zfill(7).replace(".", "")
                    start_time = float(segment[0]) / 100
                    end_time = float(segment[1]) / 100
                    data = {"AudioFile": audio, "startTime": start_time, "endTime": end_time}
                    result = lid_module(seg_key, data["AudioFile"], data["startTime"], data["endTime"])
                    f.write(seg_key+" "+result+"\n")
        f.close()
