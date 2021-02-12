
with open("../../LID/train/feats.scp") as f:
    lines = f.read().splitlines()



f2u = {}
for l in lines:
    f2u[l.split()[1]] = l.split()[0]



import pickle


with open("predict.pkl","rb") as f:
     arr = pickle.load(f)

arr.shape

len(list(f2u.keys()))

with open("idx2label","r") as f:
    lines = f.read().splitlines()

len(lines)

utts = []

for l in lines:
    utts.append(f2u[l.split()[0]])

len(utts)

from kaldiio import WriteHelper



with WriteHelper('ark,scp:/home/gnani/VGG-Speaker-Recognition/src/resvector.ark,/home/gnani/VGG-Speaker-Recognition/src/resvector.scp') as writer:
    for i in range(arr.shape[0]):
        writer(utts[i],arr[i,:])

  run.pl subsets/logs/compute_mean.log \
    ivector-mean scp:resvector.scp \
    subsets/mean.vec

filter_scp.pl resvector.scp ../../LID/train/utt2lang > subsets/utt2lang

lda_dim=150

  run.pl subsets/logs/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean subsets/mean.vec scp:resvector.scp ark:- |" \
    ark:subsets/utt2lang subsets/transform.mat


from itertools import combinations
with open("subsets/utt2lang") as f:
    lines = f.read().splitlines()

langs = set([l.split()[1] for l in lines])
combs = combinations(langs,3)
import os
for comb in combs:
    folder = "-".join(comb)
    #os.mkdir("subsets/"+folder)
    for lang in comb:
        os.system("grep "+lang+" subsets/utt2lang >> subsets/"+folder+"/utt2lang")

for comb in combs:
    folder = "-".join(comb)
    os.system("filter_scp.pl subsets/"+folder+"/utt2lang resvector.scp > subsets/"+folder+"/resvector.scp")

for comb in combs:
    folder = "-".join(comb)
    with open("subsets/"+folder+"/langs","w") as f:
        for i,lang in enumerate(comb):
            f.write(lang+" "+str(i)+"\n")
#/home/gnani/VGG-Speaker-Recognition/src/

for comb in combs:
    folder = "-".join(comb)
    model="/home/gnani/VGG-Speaker-Recognition/src/subsets/"+folder+"/logistic_regression"
    #model_rebalanced="subsets/"+folder+"/logistic_regression_rebalanced"
    languages="/home/gnani/VGG-Speaker-Recognition/src/subsets/"+folder+"/langs"
    train_utt2lang="/home/gnani/VGG-Speaker-Recognition/src/subsets/"+folder+"/utt2lang"
    train_ivectors="ark:ivector-subtract-global-mean /home/gnani/VGG-Speaker-Recognition/src/subsets/mean.vec scp:/home/gnani/VGG-Speaker-Recognition/src/subsets/"+folder+"/resvector.scp ark:- | transform-vec /home/gnani/VGG-Speaker-Recognition/src/subsets/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- | ";
    classes="ark:lid/remove_dialect.pl "+train_utt2lang+" | utils/sym2int.pl -f 2 "+languages+" - |"
    prior_scale="0.70"
    apply_log="true"
    #os.system("lid/balance_priors_to_test.pl <(lid/remove_dialect.pl <(utils/filter_scp.pl -f 1 subsets/"+folder+"/resvector.scp "+train_utt2lang)) "")
    os.system("logistic-regression-train --config=/home/gnani/VGG-Speaker-Recognition/src/logistic-regression.conf "+train_ivectors+" "+classes+" "+model+" 2>/home/gnani/VGG-Speaker-Recognition/src/subsets/"+folder+"/logistic_regression.log")







#cat run_log.sh


#!/usr/bin/env bash
# Copyright  2014   David Snyder,  Daniel Povey
# Apache 2.0.
#
# This script trains a logistic regression model on top of
# i-Vectors, and evaluates it on the NIST LRE07 closed-set
# evaluation.  

. ./cmd.sh
. ./path.sh

folder=$1

model=/home/gnani/VGG-Speaker-Recognition/src/subsets/$folder/logistic_regression
languages=/home/gnani/VGG-Speaker-Recognition/src/subsets/$folder/langs
train_utt2lang=/home/gnani/VGG-Speaker-Recognition/src/subsets/$folder/utt2lang
train_ivectors="ark:ivector-subtract-global-mean /home/gnani/VGG-Speaker-Recognition/src/subsets/mean.vec scp:/home/gnani/VGG-Speaker-Recognition/src/subsets/$folder/resvector.scp ark:- | transform-vec /home/gnani/VGG-Speaker-Recognition/src/subsets/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |";
classes="ark:lid/remove_dialect.pl $train_utt2lang | utils/sym2int.pl -f 2 $languages - |"
prior_scale=0.70
apply_log=true
logistic-regression-train --config=/home/gnani/VGG-Speaker-Recognition/src/logistic-regression.conf "$train_ivectors" "$classes" $model 2>/home/gnani/VGG-Speaker-Recognition/src/subsets/$folder/logistic_regression.log


#######################################################

import os
dirs = os.listdir("/home/gnani/VGG-Speaker-Recognition/src/subsets/")
subsets = []
for i in dirs:
    if "-" in i:
        subsets.append(i)
len(subsets)


from multiprocessing import Pool

def trigger_log(sub):
    os.system("bash run_log.sh "+sub)

def parallelizer(func, func_args):
    n_process = 48
    pool = Pool(processes=n_process)
    result = pool.map(func, func_args)
    pool.close()
    pool.join()
    return result

parallelizer(trigger_log,subsets)





