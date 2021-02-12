import os
import numpy as np


# trnlist, trnlb = toolkits.get_voxceleb2_datalist(args, path='../meta/voxlb2_train.txt')
# vallist, vallb = toolkits.get_voxceleb2_datalist(args, path='../meta/voxlb2_val.txt')


def load_from_kaldi_dir(args, part, min_len=200, label2idx=None):
    utt2num_frames = {}
    with open(os.path.join(args.data_path, part, 'utt2num_frames'), 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            utt2num_frames[l.split()[0]] = int(l.split()[1])
    utt2feat_path = {}
    with open(os.path.join(args.data_path, part, 'feats.scp'), 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            utt, feat = l.split()
            if utt2num_frames[utt] > min_len:
                utt2feat_path[utt] = feat
    if args.tandem:
        utt2post_path = {}
        with open(os.path.join(args.data_path, part, 'post.scp'), 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                utt2post_path[l.split()[0]] = l.split()[1]
    labels_file = ""
    if args.task == "lre":
        print("task is language recognition\n")
        labels_file = "utt2lang"
    elif args.task == "sre":
        print("task is speaker recognition\n")
        labels_file = "utt2spk"
    utt2label = {}
    with open(os.path.join(args.data_path, part, labels_file), 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            utt2label[l.split()[0]] = l.split()[1]
    labels = list(set(utt2label.values()))
    print("all labels found were \n", labels, '\n')
    if not label2idx:
        label2idx = {}
        for i, l in enumerate(labels):
            label2idx[l] = i
    print("Mapping: \n", label2idx)
    if not args.tandem:
        data = ([],[])
        for i in utt2feat_path.keys():
            data[0].append(utt2feat_path[i])
            data[1].append(label2idx[utt2label[i]])
        print("head(X):",data[0][:4],"\nhead(Y):",data[1][:4])
        return np.array(data[0]), np.array(data[1]), label2idx
    else:
        labels = []
        data = []
        for i in utt2feat_path.keys():
            data.append((utt2feat_path[i],utt2post_path[i]))
            labels.append(label2idx[utt2label[i]])
        print("head(X):",data[:4],"\nhead(Y):",labels[:4])
        return data, np.array(labels), label2idx


def initialize_GPU(args):
    # Initialize GPUs
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def get_chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def debug_generator(generator):
    import cv2
    import pdb
    G = generator.next()
    for i, img in enumerate(G[0]):
        path = '../sample/{}.jpg'.format(i)
        img = np.asarray(img[:, :, ::-1] + 128.0, dtype='uint8')
        cv2.imwrite(path, img)


# set up multiprocessing
def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


# vggface2 dataset
def get_vggface2_imglist(args):
    def get_datalist(s):
        file = open('{}'.format(s), 'r')
        datalist = file.readlines()
        imglist = []
        labellist = []
        for i in datalist:
            linesplit = i.split(' ')
            imglist.append(linesplit[0])
            labellist.append(int(linesplit[1][:-1]))
        return imglist, labellist

    print('==> calculating image lists...')
    # Prepare training data.
    imgs_list_trn, lbs_list_trn = get_datalist(args.trn_meta)
    imgs_list_trn = [os.path.join(args.data_path, i) for i in imgs_list_trn]
    imgs_list_trn = np.array(imgs_list_trn)
    lbs_list_trn = np.array(lbs_list_trn)

    # Prepare validation data.
    imgs_list_val, lbs_list_val = get_datalist(args.val_meta)
    imgs_list_val = [os.path.join(args.data_path, i) for i in imgs_list_val]
    imgs_list_val = np.array(imgs_list_val)
    lbs_list_val = np.array(lbs_list_val)

    return imgs_list_trn, lbs_list_trn, imgs_list_val, lbs_list_val


def get_imagenet_imglist(args, trn_meta_path='', val_meta_path=''):
    with open(trn_meta_path) as f:
        strings = f.readlines()
        trn_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        trn_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()

    with open(val_meta_path) as f:
        strings = f.readlines()
        val_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        val_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return trn_list, trn_lb, val_list, val_lb


def get_voxceleb2_datalist(args, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(args.data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return audiolist, labellist


def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def sync_model(src_model, tgt_model):
    print('==> synchronizing the model weights.')
    params = {}
    for l in src_model.layers:
        params['{}'.format(l.name)] = l.get_weights()

    for l in tgt_model.layers:
        if len(l.get_weights()) > 0:
            l.set_weights(params['{}'.format(l.name)])
    return tgt_model
