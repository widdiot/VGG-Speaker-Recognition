# Third Party
# import librosa
import numpy as np
#from kaldi.matrix import Matrix
# import time as timelib
# import scipy
# import soundfile as sf
# import scipy.signal as sps
# from scipy import interpolate
import kaldiio
#from kaldi.transform.cmvn import Cmvn

# ===============================================
#       code from Arsha for loading data.
# ===============================================
# def load_wav_fast(vid_path, sr, mode='train'):
#     """load_wav() is really slow on this version of librosa.
#     load_wav_fast() is faster but we are not ensuring a consistent sampling rate"""
#     wav, sr_ret = sf.read(vid_path)
#
#     if mode == 'train':
#         extended_wav = np.append(wav, wav)
#         if np.random.random() < 0.3:
#             extended_wav = extended_wav[::-1]
#         return extended_wav
#     else:
#         extended_wav = np.append(wav, wav[::-1])
#         return extended_wav
#
#
# def load_wav(vid_path, sr, mode='train'):
#     wav, sr_ret = librosa.load(vid_path, sr=sr)
#     assert sr_ret == sr
#
#     if mode == 'train':
#         extended_wav = np.append(wav, wav)
#         if np.random.random() < 0.3:
#             extended_wav = extended_wav[::-1]
#         return extended_wav
#     else:
#         extended_wav = np.append(wav, wav[::-1])
#         return extended_wav
#
#
# def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
#     linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
#     return linear.T
#
#
# def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
#     wav = load_wav(path, sr=sr, mode=mode)
#     linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
#     mag, _ = librosa.magphase(linear_spect)  # magnitude
#     mag_T = mag.T
#     freq, time = mag_T.shape
#     if mode == 'train':
#         if time > spec_len:
#             randtime = np.random.randint(0, time - spec_len)
#             spec_mag = mag_T[:, randtime:randtime + spec_len]
#         else:
#             spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
#     else:
#         spec_mag = mag_T
#     # preprocessing, subtract mean, divided by time-wise var
#     mu = np.mean(spec_mag, 0, keepdims=True)
#     std = np.std(spec_mag, 0, keepdims=True)
#     return (spec_mag - mu) / (std + 1e-5)

def load_kaldi_feat(path, spec_len, mode='train', cmvn=None):
    feat = kaldiio.load_mat(path)
    if cmvn is not None:
        feat = feat-cmvn
    feat_T = feat.T
    freq, time = feat_T.shape
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time - spec_len)
            feat_T_trunc = feat_T[:, randtime:randtime + spec_len]
        else:
            feat_T_trunc = np.pad(feat_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        feat_T_trunc = feat_T
    # no need for normalization because, we perform normalization and vad in kaldi
    #print("feat_T_trunc:",feat_T_trunc.shape)
    return feat_T_trunc

def load_kaldi_feat_tandem(paths, spec_len, mode='train',cmvn=None,postcmvn=None):
    path = paths[0]    #mfcc
    phone_path = paths[1]    #posteriors
    #print(path,phone_path)
    feat = kaldiio.load_mat(path)
    if cmvn is not None:
        feat = feat-cmvn
    feat_T = feat.T
    freq, time = feat_T.shape
    post = kaldiio.load_mat(phone_path)
    if postcmvn is not None:
        post = post-postcmvn
    post_T = post.T
    phones, ptime = post_T.shape
 #       assert ptime == time, "phone length and mfcc length must be same: mfcc= %s, post=%s" %(path, phone_path)
    if ptime != time:
        time = min(ptime,time)
        feat_T = feat_T[:,:time]
        post_T = post_T[:,:time]
            
    assert feat_T.shape[1] == post_T.shape[1], "phone length and mfcc length must be same: mfcc= %s, post=%s" %(path, phone_path)
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time - spec_len)
            feat_T_trunc = feat_T[:, randtime:randtime + spec_len]
            post_T_trunc = post_T[:, randtime:randtime + spec_len]
            tandem_feat = np.concatenate((feat_T_trunc, post_T_trunc), axis=0)
        else:
            feat_T_trunc = np.pad(feat_T, ((0, 0), (0, spec_len - time)), 'constant')
            post_T_trunc = np.pad(post_T, ((0, 0), (0, spec_len - time)), 'constant')
            tandem_feat = np.concatenate((feat_T_trunc, post_T_trunc), axis=0)
    else:
        feat_T_trunc = feat_T
        post_T_trunc = post_T
        tandem_feat = np.concatenate(feat_T_trunc, post_T_trunc, axis=0)
    # no need for normalization because, we perform normalization and vad in kaldi
    #print("feat_T_trunc:",feat_T_trunc.shape)
    return tandem_feat

if __name__ == "__main__":
    paths = ('/home/gnani/LID/mfcc_preproc/xvector_feats_FULL_mfcc.1.ark:67380979', '/home/gnani/LID/phone_posteriors/post.ark:292372205')
    print(load_kaldi_feat_tandem(paths, 500).shape)
