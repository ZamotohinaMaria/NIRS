# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_relgan.py
# @Time         : Created at 2019-05-28
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import sys
from subprocess import call

import os

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = 0
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    job_id = 0
    gpu_id = 0
    print('Missing argument: job_id and gpu_id. Use default job_id: {}, gpu_id: {}'.format(job_id, gpu_id))

# Executables
executable = 'python'  # specify your own python interpreter path here
rootdir = '../'
scriptname = 'main.py'

# ===Program===
if_test = int(False)
run_model = 'relgan'
CUDA = int(True)
oracle_pretrain = int(True)
gen_pretrain = int(False)
dis_pretrain = int(False)
MLE_train_epoch = 300 #300
ADV_train_epoch = 250 #1500
tips = 'RelGAN high-fidelity experiments ({})'

# ===Real data===
# job_id=0 -> goodware, job_id=1 -> malware
if_real_data = [int(True), int(True)]
dataset = ['MalbehavD-V1_goodware', 'MalbehavD-V1_malware']
class_name = ['goodware', 'malware']
loss_type = 'rsgan'
vocab_size = [0, 0]
temp_adpt = 'exp'
temperature = [40, 40]

# ===Basic Param===
data_shuffle = int(False)
model_type = 'vanilla'
gen_init = 'truncated_normal'
dis_init = 'uniform'
samples_num = 4096
batch_size = 32
max_seq_len = 20  # auto-overwritten from dataset file in main.py for real data
gen_lr = 5e-3
gen_adv_lr = 5e-5
dis_lr = 5e-5
pre_log_step = 25
adv_log_step = 50

# ===Generator===
ADV_g_step = 1
gen_embed_dim = 64
gen_hidden_dim = 64
mem_slots = 1
num_heads = 2
head_size = 128

# ===Discriminator===
ADV_d_step = 3
dis_embed_dim = 128
dis_hidden_dim = 128
num_rep = 64

# ===Metrics===
use_nll_oracle = int(False)
use_nll_gen = int(True)
use_nll_div = int(True)
use_bleu = int(True)
use_self_bleu = int(False)
use_ppl = int(False)

if job_id >= len(dataset):
    raise ValueError('job_id {} is out of range, max is {}.'.format(job_id, len(dataset) - 1))

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--cuda', CUDA,
    '--device', gpu_id,
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--tips', tips.format(class_name[job_id]),

    # Oracle or Real
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset[job_id],
    '--loss_type', loss_type,
    '--vocab_size', vocab_size[job_id],
    '--temp_adpt', temp_adpt,
    '--temperature', temperature[job_id],

    # Basic Param
    '--shuffle', data_shuffle,
    '--model_type', model_type,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--samples_num', samples_num,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--gen_adv_lr', gen_adv_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,
    '--mem_slots', mem_slots,
    '--num_heads', num_heads,
    '--head_size', head_size,

    # Discriminator
    '--adv_d_step', ADV_d_step,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,
    '--num_rep', num_rep,

    # Metrics
    '--use_nll_oracle', use_nll_oracle,
    '--use_nll_gen', use_nll_gen,
    '--use_nll_div', use_nll_div,
    '--use_bleu', use_bleu,
    '--use_self_bleu', use_self_bleu,
    '--use_ppl', use_ppl,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
