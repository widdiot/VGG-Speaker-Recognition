#!/usr/bin/env python

from __future__ import print_function

from kaldi.segmentation import NnetSAD, SegmentationProcessor
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader
import sox
import os
import time
import uuid
import hashlib
from multiprocessing import Pool
import shutil
#from pydub import AudioSegment
import json
from configparser import ConfigParser
import logging

logger = logging.getLogger(__name__)

# Reading from config file
parser = ConfigParser()
parser.read('sad_model.conf')
samplerate = int(parser.get('AUDIO', 'SAMPLE_RATE'))
n_channels = int(parser.get('AUDIO', 'N_CHANNELS'))
encoding = parser.get('AUDIO', 'ENCODING')
sad_final_raw = parser.get('SAD', 'FINAL_RAW')
post_output_vec = parser.get('SAD', 'POST_OUTPUT_VEC')

# Construct SAD
model = NnetSAD.read_model(sad_final_raw)
post = NnetSAD.read_average_posteriors(post_output_vec)
transform = NnetSAD.make_sad_transform(post)
graph = NnetSAD.make_sad_graph(min_silence_duration=0.1)
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.extra_left_context = 79
decodable_opts.extra_right_context = 21
decodable_opts.extra_left_context_initial = 0
decodable_opts.extra_right_context_final = 0
decodable_opts.frames_per_chunk = 150
decodable_opts.acoustic_scale = 0.3
sad = NnetSAD(model, transform, graph, decodable_opts=decodable_opts)
seg = SegmentationProcessor(target_labels=[2])


def convert_to_wav(path):
    '''
        1.  Converts audio file in other formats to wav format
        2.  Output directory is same as input file directory
        3.  Returns the path of wav file with success/failure
    '''
    out_path = path.replace(path.split(".")[-1], ".wav")
    try:
        sound = AudioSegment.from_file(path, format=path.split(".")[-1])
        sound.export(out_path, format="wav")
        return out_path, True
    except Exception as e:
        logger.info("please give audio in some standard format")
        return path, False


def convert_to_mono(input_file):
    """
        1. In-place conversion of wav file to required sampling rate and channels
    """
    output_file = "/tmp/" + str(uuid.uuid4()) + ".wav"
    tfm = sox.Transformer()
    tfm.set_output_format(encoding=encoding, bits=16, rate=samplerate, channels=n_channels)
    tfm.build(input_file, output_file)
    shutil.move(output_file, input_file)


def build_response(audio_file):
    """
        1. Computes MFCC features required for SAD module
        2. Performs SAD and updates segments based on min segment duration
        3. Performs decoding sequentially
        4. Returns segment-wise response of the complete audio
    """
    utt_key = audio_file.split("/")[-1].replace(".wav", "")
    feats_rspec = "ark:compute-mfcc-feats --config=sad/mfcc_hires.conf 'scp:echo " + utt_key + " " + audio_file + " |' ark:- |"
    feats = SequentialMatrixReader(feats_rspec)
    key = feats.key()
    out = sad.segment(feats.value())
    segments, stats = seg.process(out["alignment"])
    segments = seg.merge_consecutive_segments(segments, stats)
    duration = 0
    data_dict = {}
    for segment in segments:
        start_time = float(segment[0]) / 100
        end_time = float(segment[1]) / 100
        duration += (end_time - start_time)
    return segments, duration


def parallelizer(func, func_args):
    '''
        1. Function parallelizes the task to n_process (default=2)
    '''
    n_process = 8
    pool = Pool(processes=n_process)
    result = pool.map(func, func_args)
    pool.close()
    pool.join()
    return result


def decoder_dual(audio_file):
    """
       1.  Function checks if the given audio is in wav format if not convertes into wav format
       2.  In case of wave file dual channel or mono channel wav file
       3.  Returns JSON response consisting of word level confidence, sentence level confidence and word level timestamps
    """
    check = True
    if not audio_file.endswith(".wav"):
        audio_file, check = convert_to_wav(audio_file)
    if check:
        speaker = False
        start = time.time()
        convert_to_mono(audio_file)
        result = build_response(audio_file)
        logger.info("Time Taken for whole audio:" + str(time.time() - start))
    else:
        logger.error("Error during decoding")
    f = open(".".join(audio_file.split(".")[:-1]) + ".txt", "w")
    f.write(".".join(audio_file.split(".")[:-1]) + ".wav," + str(result) + "\n")
    f.close()


if __name__ == "__main__":
    import sys
    import glob

    segments, duration = build_response(audio_file="test.wav")

    print(duration)
    print(segments)

    # if os.path.isdir(sys.argv[1]):
    #     parallelizer(decoder_dual, glob.glob(sys.argv[1] + "/*.wav"))
    #     parallelizer(decoder_dual, glob.glob(sys.argv[1] + "/*.mp3"))
    # elif os.path.isfile(sys.argv[1]):
    #     decoder_dual(sys.argv[1])
    # else:
    #     pass

