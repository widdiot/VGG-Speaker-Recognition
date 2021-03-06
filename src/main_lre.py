from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
import numpy as np
#from kaldi.transform.cmvn import Cmvn
import kaldiio
np.seterr(all='raise')

sys.path.append('../tool')
import toolkits

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--task', default='lre', choices=['lre', 'sre'], type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--tandem', default=False, type=bool)
parser.add_argument('--dim', default=40, type=int)
parser.add_argument('--data_path', default='/home/gnani/LID/', type=str)
parser.add_argument('--multiprocess', default=12, type=int)
parser.add_argument('--cmvn', default='/home/gnani/LID/train/cmvn.ark', type=str)
parser.add_argument('--post_cmvn',default='',type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax','tuplemax'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    import generator

    # ==================================
    #       Get Train/Val.
    # ==================================

    trnlist, trnlb, l2i = toolkits.load_from_kaldi_dir(args, "train", min_len=300)
    vallist, vallb, _ = toolkits.load_from_kaldi_dir(args, "val", min_len=300, label2idx=l2i)
    if args.cmvn:   
        cmvn_stats = kaldiio.load_mat(args.cmvn)
        mean_stats = cmvn_stats[0,:-1]
        count = cmvn_stats[0,-1]
        offset = np.expand_dims(mean_stats,0)/count
        print("offset",offset)
        CMVN = offset

    else:
        CMVN = None

    if args.post_cmvn:
        cmvn_stats = kaldiio.load_mat(args.post_cmvn)
        mean_stats = cmvn_stats[0,:-1]
        count = cmvn_stats[0,-1]
        offset = np.expand_dims(mean_stats,0)/count
        print("offset",offset)
        POSTCMVN = offset

    else:
        POSTCMVN = None

    # construct the data generator.
    params = {'dim': (args.dim, 300, 1),
              'mp_pooler': toolkits.set_mp(processes=args.multiprocess),
              'nfft': 512,
              'spec_len': 300,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 8,
              'sampling_rate': 16000,
              'tandem': args.tandem,
              'batch_size': args.batch_size,
              'shuffle': True,
              'normalize': False,
              'cmvn': CMVN,
              'postcmvn': POSTCMVN
              }

    # Datasets
    partition = {'train': trnlist, 'val': vallist}
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}

    # Generators
    trn_gen = generator.DataGenerator(partition['train'], labels['train'], **params)
    val_gen = generator.DataGenerator(partition['val'], labels['val'], **params)
    network = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='train', args=args)
    # ==> load pre-trained model ???
    mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())

    if args.resume:
        print("Attempting to load", args.resume)
        if args.resume:
            if os.path.isfile(args.resume):
                if mgpu == 1:
                    # by_name=True, skip_mismatch=True
                    # https://github.com/WeidiXie/VGG-Speaker-Recognition/issues/46
                    network.load_weights(os.path.join(args.resume), by_name=True, skip_mismatch=True)
                else:
                    network.layers[mgpu + 1].load_weights(os.path.join(args.resume))
                print('==> successfully loading model {}.'.format(args.resume))
            else:
                print("==> no checkpoint found at '{}'".format(args.resume))

    print(network.summary())
    print('==> gpu {} is, training {} images, classes: 0-{} '
          'loss: {}, aggregation: {}, ohemlevel: {}'.format(args.gpu, len(partition['train']), np.max(labels['train']),
                                                            args.loss, args.aggregation_mode, args.ohem_level))

    model_path, log_path = set_path(args)
    with open(os.path.join(model_path,'label2idx'),'w') as f:
        for key in l2i.keys():
            f.write(key+' '+str(l2i[key])+'\n')

    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
                                              update_freq=args.batch_size * 16)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{val_loss:.3f}.h5'),
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True),
                 normal_lr, tbcallbacks]

    if args.ohem_level > 1:     # online hard negative mining will be used
        candidate_steps = int(len(partition['train']) // args.batch_size)
        iters_per_epoch = int(len(partition['train']) // (args.ohem_level*args.batch_size))

        ohem_generator = generator.OHEM_generator(network,
                                                  trn_gen,
                                                  candidate_steps,
                                                  args.ohem_level,
                                                  args.batch_size,
                                                  params['dim'],
                                                  params['n_classes']
                                                  )

        A = ohem_generator.next()   # for some reason, I need to warm up the generator

        network.fit_generator(generator.OHEM_generator(network, trn_gen, iters_per_epoch,
                                                       args.ohem_level, args.batch_size,
                                                       params['dim'], params['n_classes']),
                              steps_per_epoch=iters_per_epoch,
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1)

    else:
        network.fit_generator(trn_gen, 
                              validation_data=val_gen,
                              steps_per_epoch=int(len(partition['train'])//args.batch_size),
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=True,
                              workers=12,
                              verbose=1)


def step_decay(epoch):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    """
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


def set_path(args):
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    if args.aggregation_mode == 'avg':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    elif args.aggregation_mode == 'vlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_'
                                'bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    elif args.aggregation_mode == 'gvlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_ghost{args.ghost_cluster}_'
                                'bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    else:
        raise IOError('==> unknown aggregation mode.')
    model_path = os.path.join('../model_tandem', exp_path)
    log_path = os.path.join('../log_tandem', exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


if __name__ == "__main__":
    main()
