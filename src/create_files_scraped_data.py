with open("files") as g:
    lines = g.read().splitlines()

######### rename ##############

import os
import uuid

for l in lines:
    new = str(uuid.uuid4())
    new_file = "/".join(l.split("/")[:-1] + [new+".wav"] )
    os.rename(l,new_file)

############3 make wav.scp, utt2spk, utt2lang##############

with open("wav.scp","w") as f:
    for l in lines:
        f.write(l.split("/")[-1][:-4]+" cat "+l+" | sox -t wav - -c 1 -r 16000 -t wav - |"+"\n")

with open("utt2spk","w") as f:
    for l in lines:
        utt = l.split("/")[-1][:-4]
        #spk = l.split("/")[-2]
        f.write(utt+" "+utt+"\n")

with open("utt2lang","w") as f:
    for l in lines:
        f.write(l.split("/")[-1][:-4]+" "+l.split("/")[-2]+"\n")
##########################################################
with open("w.scp","w") as f:
    for l in lines:
        utt,path = l.split()
        new_utt = uun[utt]
        f.write(new_utt+" "+path+"\n")

######################## run sad ############################
steps/segmentation/detect_speech_activity.sh --nj 48 --cmd run.pl data/air/ exp/segmentation_1a/tdnn_stats_asr_sad_1a/ mfcc_sad/ exp/seg/ data/air_sad

steps/segmentation/detect_speech_activity.sh --nj 48 --cmd run.pl data/news/ exp/segmentation_1a/tdnn_stats_asr_sad_1a/ mfcc_sad/ exp/seg_news/ data/news_sad

steps/segmentation/detect_speech_activity.sh --nj 48 --cmd run.pl data/extras/ exp/segmentation_1a/tdnn_stats_asr_sad_1a/ mfcc_sad/ exp/seg_extras/ data/extras_sad

steps/segmentation/detect_speech_activity.sh --nj 48 --cmd run.pl data/trell/ exp/segmentation_1a/tdnn_stats_asr_sad_1a/ mfcc_sad/ exp/seg_trell/ data/trell_sad

###################### filter less than 3sec ####################

awk '$2>=300' utt2num_frames > u2n

awk '{sum += $2} END {print sum}' u2n

cp -r trell_sad_seg/ trell_ge3sec/

mv u2n utt2num_frames

fix_data_dir.sh

#################### do music vs speech classification #################
##cat conf/mfcc.conf 
# config for high-resolution MFCC features, intended for neural network training.
# Note: we keep all cepstra, so it has the same info as filterbank features,
# but MFCC is more easily compressible (because less correlated) which is why
# we prefer this method.
--use-energy=false   # use average of log energy, not energy.
--sample-frequency=16000 #  Switchboard is sampled at 8kHz
--num-mel-bins=40     # similar to Google's setup.
--num-ceps=40     # there is no dimensionality reduction.
--low-freq=40    # low cutoff frequency for mel bins
--high-freq=-200 # high cutoff frequently, relative to Nyquist of 4000 (=3800)
--allow-downsample=true





cp -r data/trell_ge3sec/ ../../bn_music_speech/v1/data/

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/trell_ge3sec/ exp/make_mfcc mfcc_trell_16k

sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    data/bn exp/make_vad $vaddir

sid/music_id.sh --cmd "$train_cmd" --nj 40 \
  exp/full_ubm_music exp/full_ubm_speech \
  data/trell_ge3sec/ exp/trell_ge3sec/


################### make monochannel from dual #########################
>>> with open("wav.scp") as f:
...     lines = f.read().splitlines()
... 
>>> lines [0]
'0004e92e-c4cd-4597-a283-63ebb234fb2f /home/gnani/news/raw/tamil/0004e92e-c4cd-4597-a283-63ebb234fb2f.wav'
>>> with open("w.scp","w") as f:
...     for l in lines:
...         utt,wav = l.split()
...         f.write(utt+" cat "+wav+" | sox -t wav - -c 1 -t wav - |"+"\n")


