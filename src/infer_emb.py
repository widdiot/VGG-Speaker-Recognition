from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import sys

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
from kaldi.ivector import IvectorExtractor, IvectorExtractorUtteranceStats, compute_vad_energy, VadEnergyOptions, LogisticRegressionConfig, LogisticRegression
from kaldi.feat.functions import sliding_window_cmn, SlidingWindowCmnOptions
from kaldi.ivector import LogisticRegressionConfig, LogisticRegression
from kaldi.feat.functions import sliding_window_cmn, SlidingWindowCmnOptions
from kaldi.matrix import Vector, SubVector, DoubleVector, Matrix
from kaldi.util.io import xopen
from kaldi.matrix.common import MatrixTransposeType
import time
import kaldiio
import model_emb
import math
# ===========================================
#        Parse the argument
# ===========================================
import argparse


#/home/repos/gnani_lre/trell/hin_IN/b8cded16b02de3c7bece4fa64b794e01560096f8_apiservice_20201130160732215437.wa

#/home/repos/gnani_lre/trell/ben_IN/b8cded16b02de3c7bece4fa64b794e01560096f8_apiservice_20201130125658816940.wav

#/home/repos/gnani_lre/trell/tel_IN/b8cded16b02de3c7bece4fa64b794e01560096f8_apiservice_20201201055513068126.wav

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='/home/repos/VGG-Speaker-Recognition/result/8lang/weights-14-0.501.h5', type=str)
parser.add_argument('--task', default='lre', choices=['lre', 'sre'], type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--labels', default='/home/repos/VGG-Speaker-Recognition/result/8lang/label2idx', type=str)
parser.add_argument('--cmvn', default='/home/repos/VGG-Speaker-Recognition/result/spont/cmvn.ark', type=str)
parser.add_argument('--data_path', default='/home/repos/gnani_lre/trell/tel_IN/b8cded16b02de3c7bece4fa64b794e01560096f8_apiservice_20201201055513068126.wav', type=str)
# set up network configuration.
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
          'spec_len': 300,
          'win_length': 400,
          'hop_length': 160,
          'n_classes': 8,
          'sampling_rate': 16000,
          'normalize': False,
          }

network_eval = model_emb.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                            num_class=params['n_classes'],
                                            mode='eval', args=args)

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
fopts.samp_freq = 16000
fopts.snip_edges = True

hires_mb_opts = MelBanksOptions()
hires_mb_opts.low_freq = 40
hires_mb_opts.high_freq = -200
hires_mb_opts.num_bins = 40
hires_mfcc_opts = MfccOptions()
hires_mfcc_opts.frame_opts = fopts
hires_mfcc_opts.num_ceps = 40
hires_mfcc_opts.mel_opts = hires_mb_opts
hires_mfcc_opts.use_energy = False

############## LOGISTIC REGRESSION MODELS #################
language_classifier = LogisticRegression()
ki = xopen("/home/repos/VGG-Speaker-Recognition/result/telugu-hindi-bengali/logistic_regression")
language_classifier.read(ki.stream(),ki.binary)  

mean = Vector()
ki = xopen('/home/repos/VGG-Speaker-Recognition/result/telugu-hindi-bengali/mean.vec')
mean.read_(ki.stream(), ki.binary)

lda = Matrix()
ki = xopen('/home/repos/VGG-Speaker-Recognition/result/telugu-hindi-bengali/transform.mat')
lda.read_(ki.stream(), ki.binary)

##### LABELS ##########

with open("/home/repos/VGG-Speaker-Recognition/result/telugu-hindi-bengali/langs") as f:
    lines = f.read().splitlines()
i2l = {}
for l in lines:
    i2l[int(l.split()[1])] = l.split()[0]


def lid_module(key, audio_file, start, end):
    # ==================================
    #       Get data and process it.
    # ==================================
    wav_spc = "scp:echo " + key + " 'sox -V0 -t wav " + audio_file + " -c 1 -r 16000 -t wav - trim " + str(start) + " " + str(
        float(end) - float(start)) + "|' |"
    hires_mfcc = Mfcc(hires_mfcc_opts)
    wav = SequentialWaveReader(wav_spc).value()
    hi_feat = hires_mfcc.compute_features(wav.data()[0], wav.samp_freq, 1.0)
    hi_feat = hi_feat.numpy() - CMVN
    X = hi_feat.T
    X = np.expand_dims(np.expand_dims(X, 0), -1)
    v = network_eval.predict(X)
    #print(v.shape)
    v = Vector(v[0,:])
    v.add_vec_(-1.0, mean)
    rows, cols = lda.num_rows, lda.num_cols
    vec_dim = v.dim
    vec_out = Vector(150)
    vec_out.copy_col_from_mat_(lda, vec_dim)
    vec_out.add_mat_vec_(1.0, lda.range(0, rows, 0, vec_dim), MatrixTransposeType.NO_TRANS, v, 1.0)
    norm = vec_out.norm(2.0)
    ratio = norm / math.sqrt(vec_out.dim)
    vec_norm = vec_out.scale_(1.0 / ratio)
    output = language_classifier.get_log_posteriors_vector(vec_norm)
    print(i2l[output.max_index()[1]])

if __name__ == "__main__":
    #wavs = glob.glob(args.data_path + '*/*.wav')
    wavs = [args.data_path]
    for audio in wavs:
        utt = audio.split('/')[-1][:-4]
        Segments, _ = save_segs.build_response(audio)
        for segment in Segments:
            seg_key = utt + "-" + str("{0:.2f}".format(float(segment[0]) / 100)).zfill(7).replace(".", "") + "-" + str(
                "{0:.2f}".format(float(segment[1]) / 100)).zfill(7).replace(".", "")
            start_time = float(segment[0]) / 100
            end_time = float(segment[1]) / 100
            if end_time - start_time >= 3:
                data = {"AudioFile": audio, "startTime": start_time, "endTime": end_time}
                lid_module(seg_key, data["AudioFile"], data["startTime"], data["endTime"])
