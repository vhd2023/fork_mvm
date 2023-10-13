import numpy as np
import torch
import argparse

from network_multi_pos import Network
from torch.utils.tensorboard import SummaryWriter

from collections import deque

# from network_hf import Network
import loss_git
from utils.save_model import save_model, save_model_10
from torch.utils import data
import pandas as pd
# from sentence_transformers import SentenceTransformer
from EDA.augment import gen_eda
from utils import cluster_utils
import os
import itertools
import torch.nn as nn

# from tensorboard.
# import nlpaug.augmenter.word as naw  # 14, 15, 18, 23, 24, and 27
import functools
import time
from pathlib import Path
import wandb as wb
from datetime import datetime

tb_writer = None
global_step = 0
batch_counter = 0
start_itr = 0
global_check = "."
batch_loss_dicts = []
run_name = "output5555555555555555555"
run_note = ""
init_epoch_loss = None
init_batch_loss = None
metric_list = []
loss_list = []
mode = "offline"
#do_colab = False
# colab_mode = True
# t5_dir = rf"My Drive/EDA/"
# file = open(f"batch_loss_logs_mlm_mix.out", "a")
p = Path("colog").resolve()
p.mkdir(exist_ok=True)
batch_len = -1
t5t1 = -1
t5t2 = -1
# file = open(f"batch_loss_logs_mlm_mix.out", "a")


def printLog(
    *args,
    **kwargs,
):
    # global_check
    t = str(datetime.now().strftime("%H:%M"))
    print(        f"|{t}|{t5t1:>2d},{int(100 * t5t2):<2d}|b:{batch_len:<4d} Epoch:{global_step:>3d} ", *args, **kwargs)
    if mode != "disabled":
        with open(f"colog/{run_name}.out", "a") as file:
            print(f"|{t}|{t5t1:>2d},{int(100 * t5t2):<2d}|b:{batch_len:<4d} Epoch:{global_step:>3d} ", *args, **kwargs, file=file)

def printLossLog(
    *args,
    **kwargs,
):
    # global_check
    t = str(datetime.now().strftime("%H:%M"))
    print(f"|{t}[{t5t1},{t5t2:.2f}] Epoch={global_step}|", *args, **kwargs)
    if mode != "disabled":
        with open(f"LOSSSSSSSS_{run_name}.out", "a") as file:
            print(f"{t}[{t5t1},{t5t2}]| Epoch={global_step}|", *args, **kwargs, file=file)


# =============================================================================
# def printLossLog(*args, **kwargs):
#     # global_check
#     print(*args, **kwargs)
#     if mode != "disabled":
#         with open(f"{run_name}.out", "a") as file:
#             print(*args, **kwargs, file=file)
#
# =============================================================================


# def write_scaler(k, v, step):
#    writer.add_scalar(k, v, step)
#    writer.add_scalar(f"batch/{k}", v, step, new_style=True)


# timer_log = {}
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        dif = time.time() - start
        dif_min = dif / 60
        name = func.__name__

        if  global_step - start_itr < 20 or global_step % 20 == 0:
            if True or dif_min > 0.1:
                printLog(
                    f"## timer step {global_step:3d} ## {name:.10s}: {dif_min:.3f} MIN"
                )
        if global_step - start_itr < 20 or global_step % 20:
            wb.log({f"t(m)/{name}": dif_min, f"t(s)/{name}": dif})
            writer.add_scalar(f"t(m)/{name}", dif_min, global_step)
            writer.add_scalar(f"t(s)/{name}", dif, global_step)
        return results

    return wrapper


def get_args_parser():
    parser = argparse.ArgumentParser("TCL for clustering", add_help=False)
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size per GPU"
    )  # 128
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--wb_mode", default="offline", type=str)
    parser.add_argument("--aug_int", default=1200, type=int)
    parser.add_argument("--do_mlm", default=False)
    parser.add_argument("--do_mix", default=False)

    # Model parameters
    parser.add_argument(
        "--feature_dim", default=128, type=int, help="dimension of ICH"
    )
    parser.add_argument(
        "--instance_temperature",
        default=0.5,
        type=float,
        help="temperature of instance-level contrastive loss",
    )
    parser.add_argument(
        "--cluster_temperature",
        default=1.0,
        type=float,
        help="temperature of cluster-level contrastive loss",
    )

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument("--init_eval", default=True)
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=5e-6,
        help="learning rate of backbone",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=5e-4,
        help="learning rate of head",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_dir",
        default="./datasets/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--dataset",
        default="StackOverflow",
        type=str,
        help="dataset",
        choices=["StackOverflow", "Biomedical", "SearchSnippets"],
    )
    parser.add_argument(
        "--class_num",
        default=20,
        type=int,
        help="number of the clusters",
    )
    parser.add_argument("--grad_steps", default=1, type=int, help="accumulation_steps ")

    parser.add_argument(
        "--model_path",
        default="save/StackOverflow/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--contrast_count", default=4, type=int)
    parser.add_argument("--do_colab", default=True)

    parser.add_argument(
        "--resume",
        default=False,
        help="resume from checkpoint",
    )
    parser.add_argument("--check_id", type=str, default="this")
    parser.add_argument(
        "--start_epoch", default=0, type=int, help="start epoch"
    )
    parser.add_argument(
        "--save_freq", default=1, type=int, help="saving frequency"
    )
    parser.add_argument("--num_workers", default=8, type=int)  # 10

    return parser


def update_args(args, s):
    if args.dataset == "Biomedical":
        args.class_num = 20
        args.model_path = "save/Biomedical/"
    elif args.dataset == "SearchSnippets":
        args.class_num = 8
        args.model_path = "save/SearchSnippets/"
    elif args.dataset == "StackOverflow":
        args.class_num = 20
        args.model_path = "save/Stackoverflow/"
    else:
        raise NotImplementedError

    if args.check_id == "this" and args.resume == False:
        printLog("no checkpoint id provided from scratch")
        args.check_id = s
    # embed_path = Path(args.model_path).resolve() / s / "embed"
    # checkpoint_path = Path(args.model_path).resolve() / s
    # last_check_path = checkpoint_path / "last_5_epoch.tar"
    # best_check_path = checkpoint_path / "best_model_avg.tar"
    embed_path = Path(args.model_path).resolve() / args.check_id / "embed"
    checkpoint_path = Path(args.model_path).resolve() / args.check_id
    last_check_path = checkpoint_path / "last_10_epoch.tar"
    best_check_path = checkpoint_path / "best_model_avg.tar"
    i = 1
    while best_check_path.exists():
        i += 1
        name = f"best_model_avg_{i}.tar"
        best_check_path = best_check_path.parent / name
    global global_check
    global_check = checkpoint_path / "output.out"
    embed_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    return best_check_path, checkpoint_path, embed_path, last_check_path


class DatasetIterater(data.Dataset):
    def __init__(self, texta, textb):
        self.texta = texta
        self.textb = textb
        # printLog(type(texta), type(textb))
        # printLog(len(texta), len(textb))
        if len(texta) != len(textb):
            printLog(len(texta), len(textb))
            raise ValueError("mismatch at aug1, aug2 length...")

    # assert len(texta) == len(textb)

    def __getitem__(self, item):
        return self.texta[item], self.textb[item]

    def __len__(self):
        return len(self.texta)



class DatasetMultiPositive(data.Dataset):
    def __init__(self, augments_all):
        self.augments_all = augments_all
        # self.length = len(self.augments_all[0])
        for i, augs in enumerate(self.augments_all):
            # print(len(self.augments_all))
            # printLog(f"checking dataset len augs >>>{i}, n length...",i, len(augs), len(self))
            if len(augs) != len(self):
                print(len(self.augments_all))
                printLog(f"mismatch at augs >>>{i}, n length...",i, len(augs), len(self))
                raise ValueError(f"mismatch at augs >>>{i}, n length...",i, len(augs), len(self))

    # assert len(texta) == len(textb)

    def __getitem__(self, item):
        return [augs[item] for augs in self.augments_all], item

    def __len__(self):
        return len(self.augments_all[0])

class Datasetlast10Memory(data.Dataset):
    def __init__(self, augments_all, sents_all):
        self.augments_all = augments_all
        self.sents_all = sents_all
        self.choice = random.choice(list(range(len(self.sents_all))))
        global t5t1
        t5t1 = self.choice

            #list(zip(augments_all, sents_all))
        for i, augs in enumerate(self.augments_all): 
            if len(augs) != len(self):
                print(len(self.augments_all))
                printLog(f"mismatch at augs >>>{i}, n length...",i, len(augs), len(self))
                raise ValueError(f"mismatch at augs >>>{i}, n length...",i, len(augs), len(self))

    def __getitem__(self, item):
        return self.sents_all[self.choice][item], self.augments_all[self.choice][item]

    def __len__(self):
        return len(self.augments_all[0])

class MemoryDatasetMultiPositive(data.Dataset):
    def __init__(self, augments_all, z_dim, c_dim):
        self.augments_all = augments_all
        self.z_dim, self.c_dim = z_dim, c_dim
        for i, augs in enumerate(self.augments_all):
            if len(augs) != len(self):
                print(len(self.augments_all))
                printLog(f"mismatch at augs >>>{i}, n length...",i, len(augs), len(self))
                raise ValueError(f"mismatch at augs >>>{i}, n length...",i, len(augs), len(self))

        self.z_memory = None
        self.c_memory = None
    def __getitem__(self, index):
        return [augs[index] for augs in self.augments_all], index

    def __len__(self):
        return len(self.augments_all[0])

class EvalDatasetIterater(data.Dataset):
    def __init__(self, texta, label):
        self.texta = texta
        self.label = label
        assert len(texta) == len(label)

    def __getitem__(self, item):
        return self.texta[item], self.label[item]

    def __len__(self):
        return len(self.texta)


# @timer
def perform_augmentation_org(args, itr):
    data_dir = args.dataset_dir
    # aug1, aug2 = [], []
    each_aug = 20
    # num_augs = 5

    if (
        itr >= 100  # each_aug * 5
    ):  # or itr > each_aug * 600:  # itr > 250:  # each_aug * num_augs:
        t5 = -1

    #     # print("WtfucKKKKKKKKKKKKKKK?")
    #     path1 = os.path.join(data_dir, args.dataset + ".txt")
    #     path2 = os.path.join(data_dir, args.dataset + "ED2000A_aug2.txt")
    #     # = os.path.join(data_dir, args.dataset + "ED2000A_aug2.txt")
    else:
        t5 = itr // each_aug + 1
        if t5 > 5:
            printLog(itr, each_aug, itr // each_aug + 1, t5, sep="\nERROR:")
            t5 = -1
    # if itr % 30 == 0 or itr % each_aug in [0, 1, 2]:  # or itr % 50 == 1:
    #    printLog("T5 version: ===>", t5)
    #     # print("t5:\t", t5)
    #     path1 = Path(data_dir).resolve() / f"{args.dataset}_t5_{t5}.txt"
    #     # print(path1)
    #     path2 = os.path.join(data_dir, args.dataset + "EDA_t5_aug2.txt")
    # path4 = os.path.join(data_dir, args.dataset + ".txt")
    # t5 = 1
    aug1, aug2 = [], []
    # l = [1, 2, 3, 4, 5, -1]
    global t5t1, t5t2
    t5t2 = t5
    t5t1 = 0.2
    l = [t5]
    for t5 in l:
        sents = []
        if t5 == 0:
            path = Path(data_dir).resolve() / (args.dataset + ".txt")
        else:
            path = Path(data_dir).resolve() / f"{args.dataset}_t5_{t5}.txt"
        
        sents = [s.strip("\n") for s in open(str(path), "r", encoding="utf8")]
        aug1 = gen_eda(sents, 0.2, 0.2, 0.2, 0.2, 1)

        # EDA augmentation
        # aug1 = gen_eda(sents, 0.2, 0.2, 0.2, 0.2, 1)

        # with open(path2, "r", encoding="utf8") as f1:
        #    for line in f1:
        #        aug1.append(line.strip("\n"))
        #    f1.close()
        nlpaugs2, _ = load_nlpaug(t5=t5)
        aug2 = nlpaugs2
        # aug2 += load_nlpaug(t5=t5)
        if not (len(aug1) == len(aug2) == 20000):
            raise ValueError("len aug1:", len(aug1), "\nlen aug2:", len(aug2))
    if itr == 1:
        assert len(aug1) == len(aug2) == 20000 * len(l)
    return aug1, aug2  # [20000:]
    # print(len(aug1), len(aug2))


def load_nlpaug_org(t5):
    if t5 == 0:
        i = np.random.randint(low=0, high=512)
        # path = (
        #     r"C:\Users\ComInSys\Desktop\vahidi-workspace\StackOverflow-nlpaug"
        # )
        # path = r"My Drive\stackaugs"
        path = r"D:\text_clustering_paper\stackaugs" if not args.do_colab else r"../stackaugs"
        name = f"StackOverflow_nlpaug0.2_{i}.txt"
    else:  # "EDA\StackOverflow_t5_2_augs\StackOverflow_t5_2__nlp0.2_2.txt"
        i = np.random.randint(low=0, high=51)
        # path = rf"My Drive\EDA\StackOverflow_t5_{t5}_augs"
        path = rf"EDA\StackOverflow_t5_{t5}_augs" if not args.do_colab else fr"../EDA/StackOverflow_t5_{t5}_augs"
        name = f"StackOverflow_t5_{t5}__nlp0.2_{i}.txt"
    path = Path(path).resolve() / name
    aug2 = []
    with open(str(path), "r", encoding="utf8") as f:
        for line in f:
            aug2.append(line.strip("\n"))
    if t5 == 2:
        return aug2[20000:]
    return aug2

def get_original(t5):
    if t5 == 0:
        path = Path(args.dataset_dir).resolve() /  (args.dataset + ".txt")
    else:
        path = Path(args.dataset_dir).resolve() / f"{args.dataset}_t5_{t5}.txt"
    assert path.exists()
    sents = [s.strip("\n") for s in open(str(path), "r", encoding="utf8")]
    return sents

def collect8_augments():
    
    global t5t1, t5t2
    l = [1, 2, 3, 4, 5]
    t5t1 = np.random.choice(l)
    sub = list(np.random.choice(l,size=(4), replace=False))

    org_sents      = get_original(-1) 
    all_t_sents       = [get_original(i) for i in sub]
    org_context, _ = load_nlpaug(-1)
    t_contexts = [load_nlpaug(i)[0] for i in sub[:2]]

    augments_all = [org_sents, org_context, *t_contexts, *all_t_sents]
    return augments_all

def collect4_augments():
    global t5t1, t5t2
    l = [1, 2, 3, 4, 5]
    t5t1 = np.random.choice(l)

    org_sents      = get_original(-1) 
    t5_sents       = get_original(t5t1)
    org_context, _ = load_nlpaug(-1)
    t_context, _   = load_nlpaug(t5t1)

    augments_all = [org_sents, t5_sents, org_context, t_context]
    return augments_all

def collect_all_augments():
    # global t5t1, t5t2
    l = [0, 1, 2, 3, 4, 5]
    # t5t1 = np.random.choice(l)
    sents_all, augments_all = [], []
    for i in l:
        sents_all.append(get_original(i))
        augments_all.append(load_nlpaug(i)[0])
    # org_sents      = get_original(-1) 
    # t5_sents       = get_original(t5t1)
    # org_context, _ = load_nlpaug(-1)
    # t_context, _   = load_nlpaug(t5t1)

    # augments_all = [org_sents, t5_sents, org_context, t_context]
    return augments_all, sents_all

def collect2_augments():
    global t5t1, t5t2
    l = [1, 2, 3, 4, 5, 0, 0]
    t5t1 = np.random.choice(l)
    
    sents       = get_original(t5t1)
    context_augs, _   = load_nlpaug(t5t1)
    #if len(sents) != len(context_augs):
    # print(len(sents), len(context_augs))

    augments_all = [sents, context_augs]
    return augments_all

def collect_augments_org():
    global t5t1, t5t2
    augments_all = []
    l = [1, 2, 3, 4, 5]
    # ts = [np.random.choice(t_ind)]
    t5t1 = np.random.choice(l)

    org_sents = get_original(t5=-1) 
    #augments_all = [org_sents] * 8
                        # 1 static
    t5_sents  = get_original(t5t1) #[get_original(t5) for t5 in ts] # 5 static

    org_context, _  = load_nlpaug(t5=-1) 

    # #org_augs2, _  = load_nlpaug(t5=-1)                          # 1 dynamic                 # 1 dynamic
    t5_context = [load_nlpaug(t5)[0] for t5 in l]  # 5 dynamic
    # p_choice = [0.10, 0.15, 0.20, 0.25]
    # p = np.random.choice(p_choice)
    # t5t2 = p
    #org_sent_shuf  = get_shuffled(org_sents)                        # 1 dynamic
    #t5_sent_shuf   = [get_shuffled(t_sents) for t_sents in t5_sents] #5 dynamic

    #org_sent_eda  = gen_eda(org_sents, p, p, p, p, 1)
    # print(type(None))
    # t5_sents_eda = [gen_eda(t_sent, p, p, p, p, 1) for t_sent in t5_sents]

    # t5_augs_eda = [gen_eda(augs, p, p, p, p, 1) for augs in t5_context]
    #t5_augs[0], 
    # print(type(t5_sents_eda), len(t5_sents_eda), type(t5_sents_eda[0]), len(t5_sents_eda[0]))
    # print(t5_sents_eda)
    augments_all = [org_sents, t5_sents, org_context, *t5_context]
    # for i, x in enumerate(augments_all):
    #     print(type(x), type(x[0]), len(x[0]), len(x[0]), "@@@@@@@@@@@@@@@")

    return augments_all

from tqdm import tqdm


def pick_t(itr):
    global t5t1, t5t2
    cycle = 25
    cycle2 = 10
    if itr <= cycle * 5:
        t5t1 = t5t2 = itr // cycle + 1
    elif cycle * 5 < itr <= cycle * 6:
        t5t1 = t5t2 = 0
    else:
        if itr % cycle2 == 0:
            l = [1, 2, 3, 4, 5, 0]
            t5t1 = np.random.choice(l)
            t5t2 = np.random.choice(l)
    writer.add_scalar("aug/t1", t5t1, itr)
    writer.add_scalar("aug/t2", t5t2, itr)


# @timer
def perform_augmentation(args, itr):
    # data_dir = args.dataset_dir
    # aug1, aug2 = [], []
    # each_aug = 50
    # num_augs = 5

    # if (
    #     itr <= each_aug
    # ):  # or itr > each_aug * 600:  # itr > 250:  # each_aug * num_augs:
    #     t5 = -1
    # else:
    #     t5 = itr // each_aug
    # if itr % 20 == 0:  # or itr % each_aug in [0, 1, 2]:  # or itr % 50 == 1:
    #    printLog("T5 version: ===>", t5t1, t5t2)
    pick_t(itr)
    aug1, aug2 = [], []
    l = [t5t1]
    for t5 in l:
        nlpaugs1, ind = load_nlpaug(t5=t5t1)
        aug1 = nlpaugs1
        # p = 0.2
        p = 0.1
        # aug1 = gen_eda(aug1, p, p, p, p, 1)

    # l = [1, 2, 3, 4, 5, -1]
    l = [t5t2]
    for t5 in l:
        # sents = []
        # if t5 == 0:
        #     path = os.path.join(data_dir, args.dataset + ".txt")
        #     sents = [s.strip("\n") for s in open(path, "r", encoding="utf8")]
        #     aug2 += sents
        # else:
        # path = Path(data_dir).resolve() / f"{args.dataset}_t5_{t5}.txt"
        nlpaugs2, _ = load_nlpaug(t5=t5t2, pre=ind)
        aug2 = nlpaugs2
        p = 0.1
        # aug2 = gen_eda(aug2, p, p, p, p, 1)

    if itr % 10 == 0:
        assert len(aug1) == len(aug2) == 20000 * len(l)
    return aug1, aug2  # [20000:]


def load_nlpaug(t5):
    if t5 == 0:
        i = np.random.randint(low=0, high=512)
        # while i == pre:
        #     i = np.random.randint(low=0, high=512)
        # path = r"C:\Users\ComInSys\Desktop\vahidi-workspace\StackOverflow-nlpaug"
        if not args.do_colab:
            path = r"D:/text_clustering_paper/stackaugs"
        else:
            path = r"/content/drive/MyDrive/stackaugs"
        m = None
        name = f"StackOverflow_nlpaug0.2_{i}.txt"
    else:  # "EDA\StackOverflow_t5_2_augs\StackOverflow_t5_2__nlp0.2_2.txt"
#"D:\text_clustering_paper\EDA\StackOverflow_t5_2_augs\StackOverflow_t5_2__nlp0.2_4.txt"
        i = np.random.randint(low=0, high=51)
        # while i == pre:
        #     i = np.random.randint(low=0, high=51)
        if not args.do_colab:
            path = r"D:/text_clustering_paper/EDA/"
        else:
            path = r"/content/drive/MyDrive/EDA"

        m =  rf"StackOverflow_t5_{t5}_augs"
        # path = fr"..\EDA\StackOverflow_t5_{t5}_augs"
        name = f"StackOverflow_t5_{t5}__nlp0.2_{i}.txt"
    if m is None:
        path = Path(path).resolve() / name
    else:
        path = Path(path).resolve / m / name
    if not path.exists:
        print("#####################################", str(path))
        raise ValueError(str(path))
    aug2 = []
    with open(str(path), "r", encoding="utf8") as f:
        for line in f:
            aug2.append(line.strip("\n"))
    if t5 == 2:
        return aug2[20000:], i
    return aug2, i

                      
@timer
def train_epoch(data_loader, optimizer, criterion):
    model.train()
    global batch_counter, batch_loss_dicts, init_batch_loss, init_epoch_loss
    progress_bar = data_loader  # tqdm(data_loader)

    loss_epoch, instance_epoch, cluster_epoch, mlm_epoch, mixcse_epoch = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    n = len(data_loader)
    for step, (x_i, x_j) in enumerate(progress_bar):
        batch_counter += 1
        z_i, z_j, c_i, c_j, mixcse_loss, mlm_loss = model(x_i, x_j)
        c_i, c_j = nn.functional.softmax(c_i, dim=1), nn.functional.softmax(
            c_j, dim=1
        )
        loss_instance, loss_cluster = criterion.forward(
            z_i, z_j, c_i, c_j, None, None
        )
        loss = loss_cluster + loss_instance  # + (mixcse_loss + mlm_loss)
        if init_batch_loss is None:
            init_batch_loss = loss.item()
        # loss /= args.grad_steps
        loss.backward()

        if (step + 1) % args.grad_steps == 0 or (step + 1 == n):
            optimizer.step()
            optimizer_head.step()
            optimizer.zero_grad()
            optimizer_head.zero_grad()

        dic = {
            # "batch/mlm-ls": mlm_loss.item(),
            # "mixcse-ls": mixcse_loss.item(),
            "instance-ls": loss_instance.item(),
            "cluster-ls": loss_cluster.item(),
            "batch2first_lss": loss.item() / init_batch_loss,
            # "mlm-ratio": mlm_loss.item() / loss.item(),
            # "mixcse-ratio": mixcse_loss.item() / loss.item(),
            # "instance-ratio": loss_instance.item() / loss.item(),
            # "cluster-ratio": loss_cluster.item() / loss.item(),
            "batch-loss": loss.item(),
            # "batch/counter": batch_counter,
        }

        writer.add_scalars("LOSS/batch_map", dic, global_step=batch_counter)

        batch_loss_dicts.append(dic)
        wb.log(dic)
        instance_epoch += loss_instance.item()
        cluster_epoch += loss_cluster.item()
        # mlm_epoch += mlm_loss.item()
        # mixcse_epoch += mixcse_loss.item()
        loss_epoch += loss.item()
    # if step == 1000:
    #return (loss_epoch, instance_epoch, cluster_epoch, mlm_epoch, mixcse_epoch)
    if init_epoch_loss is None:
            init_epoch_loss = loss_epoch
    return (loss_epoch, instance_epoch, cluster_epoch, mlm_epoch, mixcse_epoch, loss_epoch / init_epoch_loss)


def timeit(t0, name, i):
        dif = time.time() - t0
        dif_min = dif / 60
        if (global_step - start_itr < 10 and i < 10) or i % 50: #  or 
            wb.log({f"t(m)/{name}": dif_min, f"t(s)/{name}": dif})
            writer.add_scalar(f"t(m)/{name}", dif_min, i)
            writer.add_scalar(f"t(s)/{name}", dif, i) 
        if False:
            printLog(
                        f"%%%%%%%%[timeit]%%%%%%%%%\t  batch {i:4d} ## {name:.10s}: {dif_min:.3f} MIN // {dif:.3f} SEC"
                    )
@timer        
def train_mutli_epoch(data_loader, optimizer, criterion):
    
    model.train()
    global batch_counter, batch_loss_dicts, init_batch_loss, init_epoch_loss, t5t2
    progress_bar = data_loader  # tqdm(data_loader)

    loss_epoch, instance_epoch, cluster_epoch, mlm_epoch, mixcse_epoch = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    
    n = len(data_loader)
    for step, augments in enumerate(progress_bar):
        print(len(augments), type(augments), type(augments[0]))
        print(len(augments[0]))
        batch_counter += 1
        z_reps, c_reps = [], []
        t = time.time()
        for x in augments:
            p_choice = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
            #p = np.random.choice(p_choice)
            p = np.random.uniform(low=0.01, high=0.25)
            t5t2 = p
            # print(p, x,sep='\n')
            x_eda  = gen_eda(list(x), p, p, p, p, 1)

            z, c, mix_loss, mlm_loss = model.single_forward(x_eda, pooler_type='avg')
            z_reps.append(z)
            c_reps.append(c)
        timeit(t, "t/forward_all_augs_batch", batch_counter)
        '''
            for i in range(0, len(augments), 2):
                x_i, x_j = augments[i], augments[i+1]
                z_i, z_j, c_i, c_j, mixcse_loss, mlm_loss = model(x_i, x_j)
                c_i, c_j = nn.functional.softmax(c_i, dim=1), nn.functional.softmax(
                    c_j, dim=1
                )
                z_group.append(z_i)
                z_group.append(z_j)
                c_group.append(c_i)
                c_group.append(c_j)
            # z_i, z_j, c_i, c_j, mixcse_loss, mlm_loss = model(x_i, x_j)
            # c_i, c_j = nn.functional.softmax(c_i, dim=1), nn.functional.softmax(
            #     c_j, dim=1
            # )
        '''
        # t = time.time()
        loss_instance, loss_cluster = criterion.my_multi_pos_loss(
                                                                    contrast_count=args.contrast_count,
                                                                    z_reps=z_reps,
                                                                    c_reps=c_reps
                                                                )
        # timeit(t, "t/my_mutli_loss_batch", batch_counter)

        
        loss = loss_cluster + loss_instance  # + (mixcse_loss + mlm_loss)
        # print(loss.item(), loss_instance.item(),loss_cluster.item(), sep='  **  ')
        if init_batch_loss is None:
            init_batch_loss = loss.item()
        loss /= args.grad_steps
        loss.backward()

        if (step + 1) % args.grad_steps == 0 or (step + 1 == n):
            optimizer.step()
            optimizer_head.step()
            optimizer.zero_grad()
            optimizer_head.zero_grad()

        dic = {
            "batch-loss": loss.item(),
            "batch2first": loss.item() / init_batch_loss,
            # "batch/mlm-ls": mlm_loss.item(),
            # "mixcse-ls": mixcse_loss.item(),
            "instance-ls": loss_instance.item(),
            "cluster-ls": loss_cluster.item(),
            
            # "mlm-ratio": mlm_loss.item() / loss.item(),
            # "mixcse-ratio": mixcse_loss.item() / loss.item(),
            # "instance-ratio": loss_instance.item() / loss.item(),
            # "cluster-ratio": loss_cluster.item() / loss.item(),
            
            # "batch/counter": batch_counter,
        }
        if step % 20 == 0:
            printLog(f"Batch {int(100 * (step + 1) / n):>3d}%=", {k: f"{v:.3f}" for k, v in dic.items()})
            writer.add_scalars("LOSS/batch", dic, global_step=batch_counter)

        batch_loss_dicts.append(dic)
        wb.log(dic)
        instance_epoch += loss_instance.item()
        cluster_epoch += loss_cluster.item()
        # mlm_epoch += mlm_loss.item()
        # mixcse_epoch += mixcse_loss.item()
        loss_epoch += loss.item()
    # if step == 1000:
    if init_epoch_loss is None:
        init_epoch_loss = loss_epoch
    return (loss_epoch, instance_epoch, cluster_epoch, mlm_epoch, mixcse_epoch, loss_epoch / init_epoch_loss)


def inference_pools(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    reprs_matrix = []
    # tq = tqdm(loader)
    for step, (x, y) in enumerate(loader):
        with torch.no_grad():
            clusters, reprs = model.forward_cluster_feature_return(x)
        clusters = clusters.detach()
        reprs = reprs.detach().cpu()
        feature_vector.extend(clusters.cpu().detach().numpy())
        reprs_matrix.append(reprs)
        labels = []
        for i in y:
            labels.append(int(i))
        labels_vector.extend(np.array(labels))
        if False and step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    reprs_matrix = torch.cat(reprs_matrix, dim=0).numpy()
    # print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector, reprs_matrix


@timer
def evaluation(dataset, model, device, epoch, args, best_score=None):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    # printLog("### Creating features from model ###") X, Y, reprs
    preds, labels_vector, reprs_matrix = inference_pools(
        data_loader, model, device
    )
    labels_vector = labels_vector - 1
    # printLog(np.min(X), np.min(Y))

    score, _ = cluster_utils.clustering_metric(
        labels_vector, preds, args.class_num
    )
    score["avg"] = (score["acc"] + score["nmi"]) / 2
    # score["epoch"] = epoch
    acc_dif, nmi_dif = 0.0, 0.0
    if best_score is not None:
        acc_dif, nmi_dif = (
            score["acc"] - best_score["acc"],
            score["nmi"] - best_score["nmi"],
        )
    # score["acc_dif"], score["nmi_dif"] = acc_dif, nmi_dif
    printLog(
        "### [ACC_dif:{:.2f}....NMI_dif:{:.2f}....NMI:{:.2f}....ACC:{:.2f}....Average:{:.2f}]".format(
            # epoch,
            acc_dif * 100,
            nmi_dif * 100,
            score["nmi"] * 100,
            score["acc"] * 100,
            score["avg"] * 100,
        )
    )

    printLossLog(
        "### [ACC_dif:{:.2f}....NMI_dif:{:.2f}....NMI:{:.2f}....ACC:{:.2f}....Average:{:.2f}]".format(
            # epoch,
            acc_dif * 100,
            nmi_dif * 100,
            score["nmi"] * 100,
            score["acc"] * 100,
            score["avg"] * 100,
        )
    )
    wb.log(score)
    ## for k, v in score.items():
    ##    writer.add_scalar(k, v, global_step)

    writer.add_scalars(
        main_tag="EVAL/metrics", tag_scalar_dict=score, global_step=global_step
    )
    writer.add_scalar("EVAL/ACC", score["acc"], global_step=global_step)
    writer.add_scalar("EVAL/NMI", score["nmi"], global_step=global_step)
    writer.add_scalar("EVAL/AVG", score["avg"], global_step=global_step)

    writer.add_embedding(
        tag=f"embeddings_{global_step}",
        mat=reprs_matrix,
        metadata=labels_vector.tolist(),
        global_step=global_step,
    )
    return labels_vector, reprs_matrix, score

                    
@timer        
def train_bank_epoch(data_loader, optimizer, criterion, z_banks, c_banks, use_cache):
    
    model.train()
    global batch_counter, batch_loss_dicts, init_batch_loss, init_epoch_loss, t5t2
    progress_bar = data_loader  # tqdm(data_loader)

    loss_epoch, instance_epoch, cluster_epoch, mlm_epoch, mixcse_epoch = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    
    n = len(data_loader)
    for step, batch in enumerate(progress_bar):
        # print(len(augments), type(augments), type(augments[0]))
        # print(batch)
        batch_counter += 1
        augments, index_list = batch
        z_reps, c_reps = [], []

        # t = time.time()
        for i, x in enumerate(augments):
            # p_choice = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
            p = np.random.uniform(low=0.01, high=0.26)

            t5t2 = p
            # print(p, x,sep='\n')
            x_eda  = gen_eda(list(x), p, p, p, p, 1)

            z, c, mix_loss, mlm_loss = model.single_forward(x_eda, pooler_type='avg')
            z_banks[i][index_list] = z.cpu()
            c_banks[i][index_list] = c.cpu()
            if not use_cache:
                z_reps.append(z)
                c_reps.append(c)

        if False and use_cache:
            z_i, z_j, c_i, c_j = z_banks[0], z_banks[1], c_banks[0], c_banks[1]
        else:
            z_i, z_j = z_reps
            c_i, c_j = c_reps
        
        loss_instance, loss_cluster = criterion.forward(
                                                        z_i, 
                                                        z_j, 
                                                        c_i,
                                                        c_j
                                                        )

        # timeit(t, "t/forward_all_augs_batch", batch_counter)
        # z_bank_i, c_bank_i, z_bank_j, c_bank_j = cache
        
        # t = time.time()
        # loss_instance, loss_cluster = criterion.my_multi_pos_loss(
        #                                                             contrast_count=args.contrast_count,
        #                                                             z_reps=z_reps,
        #                                                             c_reps=c_reps
        #                                                         )
         
        # timeit(t, "t/my_mutli_loss_batch", batch_counter)

        
        loss = loss_cluster + loss_instance  # + (mixcse_loss + mlm_loss)
        # print(loss.item(), loss_instance.item(),loss_cluster.item(), sep='  **  ')
        if init_batch_loss is None:
            init_batch_loss = loss.item()
        loss /= args.grad_steps
        loss.backward()

        if (step + 1) % args.grad_steps == 0 or (step + 1 == n):
            optimizer.step()
            optimizer_head.step()
            optimizer.zero_grad()
            optimizer_head.zero_grad()

        dic = {
            "batch-loss": loss.item(),
            "batch2first": loss.item() / init_batch_loss,
            # "batch/mlm-ls": mlm_loss.item(),
            # "mixcse-ls": mixcse_loss.item(),
            "instance-ls": loss_instance.item(),
            "cluster-ls": loss_cluster.item(),
            
            # "mlm-ratio": mlm_loss.item() / loss.item(),
            # "mixcse-ratio": mixcse_loss.item() / loss.item(),
            # "instance-ratio": loss_instance.item() / loss.item(),
            # "cluster-ratio": loss_cluster.item() / loss.item(),
            
            # "batch/counter": batch_counter,
        }
        if step % 5 == 0:
            printLog(f"Batch {int(100 * (step + 1) / n):>3d}%=", {k: f"{v:.3f}" for k, v in dic.items()})
            writer.add_scalars("LOSS/batch", dic, global_step=batch_counter)

        batch_loss_dicts.append(dic)
        wb.log(dic)
        instance_epoch += loss_instance.item()
        cluster_epoch += loss_cluster.item()
        # mlm_epoch += mlm_loss.item()
        # mixcse_epoch += mixcse_loss.item()
        loss_epoch += loss.item()
    # if step == 1000:
    if init_epoch_loss is None:
        init_epoch_loss = loss_epoch
    return (loss_epoch, instance_epoch, cluster_epoch, mlm_epoch, mixcse_epoch, loss_epoch / init_epoch_loss)

# scaler = torch.cuda.amp.GradScaler()

# for epoch in epochs:
#     for input, target in data:
#         optimizer0.zero_grad()
#         optimizer1.zero_grad()
#         with autocast(device_type='cuda', dtype=torch.float16):
#             output0 = model0(input)
#             output1 = model1(input)
#             loss0 = loss_fn(2 * output0 + 3 * output1, target)
#             loss1 = loss_fn(3 * output0 - 5 * output1, target)

#         # (retain_graph here is unrelated to amp, it's present because in this
#         # example, both backward() calls share some sections of graph.)
#         scaler.scale(loss0).backward(retain_graph=True)
#         scaler.scale(loss1).backward()

#         # You can choose which optimizers receive explicit unscaling, if you
#         # want to inspect or modify the gradients of the params they own.
#         scaler.unscale_(optimizer0)

#         scaler.step(optimizer0)
#         scaler.step(optimizer1)

#         scaler.update()
def train_bank_mixed_epoch(data_loader, optimizer, criterion, z_banks, c_banks, use_cache, scaler):
    
    model.train()
    global batch_counter, batch_loss_dicts, init_batch_loss, init_epoch_loss, t5t2
    progress_bar = data_loader  # tqdm(data_loader)

    loss_epoch, instance_epoch, cluster_epoch, mlm_ep, mix_ep = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    n = len(data_loader)
    for step, batch in enumerate(progress_bar):
        # print(len(augments), type(augments), type(augments[0]))
        # print(batch)
        batch_counter += 1
        augments, index_list = batch
        z_reps, c_reps = [], []
        optimizer.zero_grad()
        optimizer_head.zero_grad()
        # t = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            for i, x in enumerate(augments):
                # p_choice = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
                p = np.random.uniform(low=0.01, high=0.26)

                t5t2 = p
                x_eda  = gen_eda(list(x), p, p, p, p, 1)

                z, c, mix_loss, mlm_loss = model.single_forward(x_eda, pooler_type='avg')
                z_banks[i][index_list] = z.cpu()
                c_banks[i][index_list] = c.cpu()
                if not use_cache:
                    z_reps.append(z)
                    c_reps.append(c)

            if use_cache:
                z_i, z_j, c_i, c_j = z_banks[0], z_banks[1], c_banks[0], c_banks[1]
            else:
                z_i, z_j = z_reps
                c_i, c_j = c_reps

            device = "cpu"
            loss_instance, loss_cluster = criterion.forward1(
                                                            z_i.to(device), 
                                                            z_j.to(device), 
                                                            c_i,#.to(device),
                                                            c_j,#.to(device)
                                                        )

            
            loss = loss_cluster + loss_instance  # + (mixcse_loss + mlm_loss)
        
            if init_batch_loss is None:
                init_batch_loss = loss.item()
            loss /= args.grad_steps
        #loss.backward()
        scaler.scale(loss).backward()
        if True or (step + 1) % args.grad_steps == 0 or (step + 1 == n):
            scaler.step(optimizer)
            scaler.step(optimizer_head)
            # optimizer.step()
            # optimizer_head.step()
            scaler.update()
            

        dic = {
            "batch-loss": loss.item(),
            "batch2first": loss.item() / init_batch_loss,
            # "batch/mlm-ls": mlm_loss.item(),
            # "mixcse-ls": mixcse_loss.item(),
            "instance-ls": loss_instance.item(),
            "cluster-ls": loss_cluster.item(),
            
        }
        if step % 5 == 0:
            printLog(f"Batch {int(100 * (step + 1) / n):>3d}%=", {k: f"{v:.3f}" for k, v in dic.items()})
            writer.add_scalars("LOSS/batch", dic, global_step=batch_counter)

        batch_loss_dicts.append(dic)
        wb.log(dic)
        instance_epoch += loss_instance.item()
        cluster_epoch += loss_cluster.item()
        # mlm_epoch += mlm_loss.item()
        # mixcse_epoch += mixcse_loss.item()
        loss_epoch += loss.item()
    # if step == 1000:
    if init_epoch_loss is None:
        init_epoch_loss = loss_epoch
    return (loss_epoch, instance_epoch, cluster_epoch, mlm_ep, mix_ep, loss_epoch / init_epoch_loss)


def train_last10_epoch(data_loader, optimizer, criterion, z_qus, c_qus, scaler, max_memory):
    
    model.train()
    global batch_counter, batch_loss_dicts, init_batch_loss, init_epoch_loss, t5t2
    # progress_bar = data_loader  # tqdm(data_loader)

    loss_epoch, instance_epoch, cluster_epoch, mlm_ep, mix_ep = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    n = len(data_loader)
    for step, batch in enumerate(data_loader):
        # print(len(augments), type(augments), type(augments[0]))
        # print(batch)
        batch_counter += 1
        augments = batch
        choice = random.choice(list(range(6)))
        global t5t1
        t5t1 = choice
        data_loader.dataset.choice = choice
        # z_reps, c_reps = [], []
        
        # t = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            for i, x in enumerate(augments):
                p = np.random.choice([0.01, 0.05, 0.10, 0.15, 0.20, 0.25])
                # p = np.random.uniform(low=0.01, high=0.26)

                t5t2 = p
                x_eda  = gen_eda(list(x), p, p, p, p, 1)

                z, c, mix_loss, mlm_loss = model.single_forward(x_eda, pooler_type='avg')
                z_qus[i].append(z.cpu())
                c_qus[i].append(c.cpu())
                # if not use_cache:
                #     z_reps.append(z)
                #     c_reps.append(c)

            # if use_cache:
            #     z_i, z_j, c_i, c_j = z_banks[0], z_banks[1], c_banks[0], c_banks[1]
            # else:
            #     z_i, z_j = z_reps
            #     c_i, c_j = c_reps
           
            z_i = torch.cat(list(z_qus[0]), dim=0).to('cuda')
            z_j = torch.cat(list(z_qus[1]), dim=0).to('cuda')
            c_i = torch.cat(list(c_qus[0]), dim=0).to('cuda')
            c_j = torch.cat(list(c_qus[1]), dim=0).to('cuda')

            # z_i = z_qus[0][-1].to('cuda')
            # z_j = z_qus[1][-1].to('cuda')
            # c_i = c_qus[0][-1].to('cuda')
            # c_j = c_qus[1][-1].to('cuda')
            # print(z_i.shape, z_j.shape, c_i.shape, c_j.shape)

            global batch_len
            batch_len = z_i.shape[0]


            #device = "cpu"
            loss_instance, loss_cluster = criterion.forward(
                                                            z_i,    #.to(device), 
                                                            z_j,    #.to(device), 
                                                            c_i,    #.to(device),
                                                            c_j,    #.to(device)
                                                        )

            
            loss = loss_cluster + loss_instance                     # + (mixcse_loss + mlm_loss)
        
            if init_batch_loss is None:
                init_batch_loss = loss.item()
            loss /= args.grad_steps

        #loss.backward()
        optimizer.zero_grad()
        optimizer_head.zero_grad()
        scaler.scale(loss).backward(retain_graph=True)
        
        if True or (step + 1) % args.grad_steps == 0 or (step + 1 == n):
            scaler.step(optimizer)
            scaler.step(optimizer_head)
            # optimizer.step()
            # optimizer_head.step()
            scaler.update()
        # for qu in [*z_qus, *c_qus]:
        #     qu = deque(map(lambda x: x.detach(), qu), maxlen=max_memory)
            
        dic = {
            "batch-loss": loss.item(),
            "batch2first": loss.item() / init_batch_loss,
            # "batch/mlm-ls": mlm_loss.item(),
            # "mixcse-ls": mixcse_loss.item(),
            "instance-ls": loss_instance.item(),
            "cluster-ls": loss_cluster.item(), 
        }
        if step % 100 == 0: # {int(100 * (step + 1) / n):<2d}%
            printLog(f"itr {step + 1:3d}/{n} |", {k: f"{v:.3f}" for k, v in dic.items()})
            writer.add_scalars("LOSS/batch", dic, global_step=batch_counter)

        batch_loss_dicts.append(dic)
        wb.log(dic)
        instance_epoch += loss_instance.item()
        cluster_epoch += loss_cluster.item()
        # mlm_epoch += mlm_loss.item()
        # mixcse_epoch += mixcse_loss.item()
        loss_epoch += loss.item()
    # if step == 1000:
    if init_epoch_loss is None:
        init_epoch_loss = loss_epoch
    return (loss_epoch, instance_epoch, cluster_epoch, mlm_ep, mix_ep, loss_epoch / init_epoch_loss)


def train_last10_all(best_score):
    global global_step, metric_list, loss_list
    save_memory_tensor = lambda t, name: None

    
    max_memory = 12
    z_qus = (deque(maxlen=max_memory), deque(maxlen=max_memory))
    c_qus = (deque(maxlen=max_memory), deque(maxlen=max_memory))
    
    scaler = torch.cuda.amp.GradScaler()

    augments_all, sents_all = collect_all_augments() #collect4_augments() if args.contrast_count == 4 else collect8_augments()
    print(len(augments_all), [len(x) for x in augments_all], [len(x) for x in sents_all])

    # augments_all = list(np.random.permutation(augments_all))

    dataset = Datasetlast10Memory(augments_all, sents_all)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler = ShuffledSampler(data_source=dataset, batch_size=args.batch_size, n_last_batches=max_memory),
        #shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    aug_type = "external"
    printLog("from epoch {} to epoch {}".format(args.start_epoch, args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        global_step = epoch + 1
        # cache = (z_bank_i, c_bank_i, z_bank_j, c_bank_j)
        # use_cache = epoch > 0
        if epoch > 0:
            augments_all, sents_all = collect_all_augments() #collect4_augments() if args.contrast_count == 4 else collect8_augments()
            # augments_all = list(np.random.permutation(augments_all))
            #dataset = Datasetlast10Memory(augments_all, sents_all)
            data_loader.dataset.augments_all = augments_all
            data_loader.dataset.sents_all    = sents_all

            data_loader.sampler.data_source = data_loader.dataset
        
        
        with torch.autograd.set_detect_anomaly(True):
            (
                loss_epoch,
                instance_epoch,
                cluster_epoch,
                mlm_epoch,
                mixcse_epoch,
                epoch2first_lss
            ) = train_last10_epoch(data_loader, optimizer, criterion, z_qus, c_qus, scaler=scaler, max_memory=max_memory)
        
        
        dic = {
            "loss_epoch": loss_epoch,
            "epoch2first_lss": epoch2first_lss,
            "instance_lss": instance_epoch,
            "cluster_lss": cluster_epoch,

            # "mlm_lss": mlm_epoch,
            # "mixcse_lss": mixcse_epoch,
        }
        # use_cache = epoch > 0
        printLog(f"Epoch {epoch + 1} Loss:", {k: f"{v:.3f}" for k, v in dic.items()})

        # if global_step % 5 == 0:
        #     printLog(f"\t\t\t\t Loss:{loss_epoch:.3f}")
        # writer.add_scalars("epoch", tag_scalar_dict=dic, global_step=epoch + 1)
        writer.add_scalar("LOSS/sum", loss_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mlm", mlm_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mixcse", mixcse_epoch, global_step=epoch + 1)
        writer.add_scalar(
            "LOSS/instance", instance_epoch, global_step=epoch + 1
        )
        writer.add_scalar("LOSS/cluster", cluster_epoch, global_step=epoch + 1)

        loss_list.append(dic)
        wb.log(dic)

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            best = False
            labels, reprs, score = evaluation(
                dataset=evaldataset,
                model=model,
                device="cuda",
                epoch=epoch + 1,
                args=args,
                best_score=best_score,
            )
            metric_list.append(score)
            if score["avg"] > best_score["avg"]:
                best = True
                best_score = score.copy()
                dic = {f"best/{k}": v for k, v in best_score.items()}
                # metric_list.append(best)
                wb.log(dic)

                printLog(f"/\/\/\ new best at epoch {epoch + 1} /\/\/\ ")
            # printLog(dic)

            e_path = embed_path / f"iter_{epoch + 1}_embeds.npy"
            # l_path = embed_path / f"iter_{epoch + 1}_labels.npy"
            np.save(file=e_path.__str__(), arr=reprs)
            np.save(file=e_path.__str__(), arr=labels)
            if True:
                save_model(
                    args,
                    model,
                    optimizer,
                    optimizer_head,
                    epoch + 1,
                    path=checkpoint_path,
                    id=run.id,
                    best=best,
                )

        if (epoch + 1) % 1 == 0:
            save_model_10(
                args,
                model,
                optimizer,
                optimizer_head,
                epoch + 1,
                path=checkpoint_path,
                id=run.id,
            )
    return best_score



import random

class ShuffledSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, n_last_batches=1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.n_last_batches = n_last_batches

    def __iter__(self):
        t = time.time()
        printLog("Start iter shuffle sampler =====>", flush=True, end='\t')
        indices = np.random.permutation(len(self.data_source))

        # Calculate the number of batches and the size of the last batch
        num_batches, last_batch_size = divmod(len(indices), self.batch_size)

        non_overlapping_batches = np.array_split(indices[:num_batches * self.batch_size], num_batches)

        # Handle the last n batches
        for i in range(-self.n_last_batches, 0):
            start = num_batches * self.batch_size + i * self.batch_size
            end = start + self.batch_size

            # Check if there are enough samples for the last batch
            if end <= len(indices):
                non_overlapping_batches.append(indices[start:end])
        printLog("END: iter function: done>>>>", time.time() - t)

        return iter([item for sublist in non_overlapping_batches for item in sublist])

    def __len__(self):
        return len(self.data_source)

'''
import random
class ShuffledSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, n_last_batches=1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.n_last_batches = n_last_batches
        printLog("Creating shuffle sampler object")
    @timer
    def __iter__(self):
        t = time.time()
        printLog("Start iter shuffle sampler =====>", flush=True, end='\t')
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        
        # Split the shuffled indices into batches
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        # Make sure the last n batches do not have identical samples
        printLog("Before iter loop:::::", flush=True, end="\t")
        for i in range(1, self.n_last_batches):
            last_n_batches = batches[-i:]

            for batch in last_n_batches:
                # Randomly shuffle the batch until it's different from previous batches
                while any(item in prev_batch for prev_batch in last_n_batches[:-1] for item in batch):
                    random.shuffle(batch)

        iterator = iter([item for sublist in batches for item in sublist])
        printLog("END: iter function: done>>>>", time.time() - t)

        # Flatten the list of batches and create an iterator
        return iterator

    def __len__(self):
        return len(self.data_source)
'''

def train_bank_all(best_score):
    global global_step, metric_list, loss_list
    save_memory_tensor = lambda t, name: None
    n = 20000   # TODO TODO TODO
    z_dim = args.feature_dim
    c_dim = args.class_num
    z_bank_i = torch.zeros(n, z_dim)#.to('cuda')
    c_bank_i = torch.zeros(n, c_dim)#.to('cuda')
    z_bank_j = torch.zeros(n, z_dim)#.to('cuda')
    c_bank_j = torch.zeros(n, c_dim)#.to('cuda')

    z_banks = [z_bank_i, z_bank_j]
    c_banks = [c_bank_i, c_bank_j]

    scaler = torch.cuda.amp.GradScaler()

    aug_type = "external"
    printLog("from epoch {} to epoch {}".format(args.start_epoch, args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        global_step = epoch + 1
        # cache = (z_bank_i, c_bank_i, z_bank_j, c_bank_j)
        use_cache = epoch > 0
       
        augments_all = collect2_augments() #collect4_augments() if args.contrast_count == 4 else collect8_augments()
        if epoch == args.start_epoch:
            print(len(augments_all), [len(x) for x in augments_all], [type(x) for x in augments_all])

        # augments_all = list(np.random.permutation(augments_all))

        dataset = DatasetMultiPositive(augments_all)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

        with torch.autograd.set_detect_anomaly(True):
            (
                loss_epoch,
                instance_epoch,
                cluster_epoch,
                mlm_epoch,
                mixcse_epoch,
                epoch2first_lss
            ) = train_bank_mixed_epoch(data_loader, optimizer, criterion, z_banks, c_banks, use_cache=use_cache, scaler=scaler)
        
        
        dic = {
            "loss_epoch": loss_epoch,
            "epoch2first_lss": epoch2first_lss,
            "instance_lss": instance_epoch,
            "cluster_lss": cluster_epoch,

            # "mlm_lss": mlm_epoch,
            # "mixcse_lss": mixcse_epoch,
        }
        use_cache = epoch > 0
        printLog(f"Epoch {epoch + 1} Loss:", {k: f"{v:.3f}" for k, v in dic.items()})

        # if global_step % 5 == 0:
        #     printLog(f"\t\t\t\t Loss:{loss_epoch:.3f}")
        # writer.add_scalars("epoch", tag_scalar_dict=dic, global_step=epoch + 1)
        writer.add_scalar("LOSS/sum", loss_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mlm", mlm_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mixcse", mixcse_epoch, global_step=epoch + 1)
        writer.add_scalar(
            "LOSS/instance", instance_epoch, global_step=epoch + 1
        )
        writer.add_scalar("LOSS/cluster", cluster_epoch, global_step=epoch + 1)

        loss_list.append(dic)
        wb.log(dic)

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            best = False
            labels, reprs, score = evaluation(
                dataset=evaldataset,
                model=model,
                device="cuda",
                epoch=epoch + 1,
                args=args,
                best_score=best_score,
            )
            metric_list.append(score)
            if score["avg"] > best_score["avg"]:
                best = True
                best_score = score.copy()
                dic = {f"best/{k}": v for k, v in best_score.items()}
                # metric_list.append(best)
                wb.log(dic)

                printLog(f"/\/\/\ new best at epoch {epoch + 1} /\/\/\ ")
            # printLog(dic)

            e_path = embed_path / f"iter_{epoch + 1}_embeds.npy"
            # l_path = embed_path / f"iter_{epoch + 1}_labels.npy"
            np.save(file=e_path.__str__(), arr=reprs)
            np.save(file=e_path.__str__(), arr=labels)
            if True:
                save_model(
                    args,
                    model,
                    optimizer,
                    optimizer_head,
                    epoch + 1,
                    path=checkpoint_path,
                    id=run.id,
                    best=best,
                )

        if (epoch + 1) % 1 == 0:
            save_model_10(
                args,
                model,
                optimizer,
                optimizer_head,
                epoch + 1,
                path=checkpoint_path,
                id=run.id,
            )
    return best_score


def train_mutli_all(best_score):
    global global_step, metric_list, loss_list
    

    aug_type = "external"
    printLog("from epoch {} to epoch {}".format(args.start_epoch, args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        global_step = epoch + 1

       
        augments_all = collect4_augments() if args.contrast_count == 4 else collect8_augments()
        if epoch == args.start_epoch:
            print(len(augments_all), [len(x) for x in augments_all], [type(x) for x in augments_all])

        # augments_all = list(np.random.permutation(augments_all))

        dataset = DatasetMultiPositive(augments_all)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

        with torch.autograd.set_detect_anomaly(True):
            (
                loss_epoch,
                instance_epoch,
                cluster_epoch,
                mlm_epoch,
                mixcse_epoch,
                epoch2first_lss
            ) = train_mutli_epoch(data_loader, optimizer, criterion)
        dic = {
            "loss_epoch": loss_epoch,
            "epoch2first_lss": epoch2first_lss,
            "instance_lss": instance_epoch,
            "cluster_lss": cluster_epoch,

            # "mlm_lss": mlm_epoch,
            # "mixcse_lss": mixcse_epoch,
        }
        printLog(f"Epoch {epoch + 1} Loss:", {k: f"{v:.3f}" for k, v in dic.items()})

        # if global_step % 5 == 0:
        #     printLog(f"\t\t\t\t Loss:{loss_epoch:.3f}")
        # writer.add_scalars("epoch", tag_scalar_dict=dic, global_step=epoch + 1)
        writer.add_scalar("LOSS/sum", loss_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mlm", mlm_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mixcse", mixcse_epoch, global_step=epoch + 1)
        writer.add_scalar(
            "LOSS/instance", instance_epoch, global_step=epoch + 1
        )
        writer.add_scalar("LOSS/cluster", cluster_epoch, global_step=epoch + 1)

        loss_list.append(dic)
        wb.log(dic)

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            best = False
            labels, reprs, score = evaluation(
                dataset=evaldataset,
                model=model,
                device="cuda",
                epoch=epoch + 1,
                args=args,
                best_score=best_score,
            )
            metric_list.append(score)
            if score["avg"] > best_score["avg"]:
                best = True
                best_score = score.copy()
                dic = {f"best/{k}": v for k, v in best_score.items()}
                # metric_list.append(best)
                wb.log(dic)

                printLog(f"/\/\/\ new best at epoch {epoch + 1} /\/\/\ ")
            # printLog(dic)

            e_path = embed_path / f"iter_{epoch + 1}_embeds.npy"
            # l_path = embed_path / f"iter_{epoch + 1}_labels.npy"
            np.save(file=e_path.__str__(), arr=reprs)
            np.save(file=e_path.__str__(), arr=labels)
            if True:
                save_model(
                    args,
                    model,
                    optimizer,
                    optimizer_head,
                    epoch + 1,
                    path=checkpoint_path,
                    id=run.id,
                    best=best,
                )

        if (epoch + 1) % 1 == 0:
            save_model_10(
                args,
                model,
                optimizer,
                optimizer_head,
                epoch + 1,
                path=checkpoint_path,
                id=run.id,
            )
    return best_score

def train_all(best_score):
    global global_step, metric_list, loss_list

    
    aug_type = "external"
    printLog("from epoch {} to epoch {}".format(args.start_epoch, args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        global_step = epoch + 1

        # prepare data
        # data_dir = args.dataset_dir
        # print("### start augmentation")
        if aug_type == "external":
            aug1, aug2 = perform_augmentation_org(args=args, itr=epoch + 1)
        else:
            raise ValueError("internal????")
            aug1 = sents
            aug2 = sents.copy()
        dataset = DatasetIterater(aug1, aug2)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

        # data_loader = torch.utils.data.DataLoader(
        # dataset,
        # batch_size=args.batch_size,
        # shuffle=True,
        # drop_last=True,
        # num_workers=args.num_workers
        # )

        (
            loss_epoch,
            instance_epoch,
            cluster_epoch,
            mlm_epoch,
            mixcse_epoch,
            epoch2first_lss
        ) = train_epoch(data_loader, optimizer, criterion)
        dic = {
            "loss_epoch": loss_epoch,
            "epoch2first_lss": epoch2first_lss,

            "instance_lss": instance_epoch,
            "cluster_lss": cluster_epoch,
            # "mlm_lss": mlm_epoch,
            # "mixcse_lss": mixcse_epoch,
        }

        if global_step % 5 == 0:
            printLog(f"\t\t\t\t Loss:{loss_epoch:.3f}")
        # writer.add_scalars("epoch", tag_scalar_dict=dic, global_step=epoch + 1)
        writer.add_scalar("LOSS/sum", loss_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mlm", mlm_epoch, global_step=epoch + 1)
        # writer.add_scalar("LOSS/mixcse", mixcse_epoch, global_step=epoch + 1)
        writer.add_scalar(
            "LOSS/instance", instance_epoch, global_step=epoch + 1
        )
        writer.add_scalar("LOSS/cluster", cluster_epoch, global_step=epoch + 1)

        loss_list.append(dic)
        wb.log(dic)

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            best = False
            labels, reprs, score = evaluation(
                dataset=evaldataset,
                model=model,
                device="cuda",
                epoch=epoch + 1,
                args=args,
                best_score=best_score,
            )
            metric_list.append(score)
            if score["avg"] > best_score["avg"]:
                best = True
                best_score = score.copy()
                dic = {f"best/{k}": v for k, v in best_score.items()}
                # metric_list.append(best)
                wb.log(dic)

                printLog(f"/\/\/\ new best at epoch {epoch + 1} /\/\/\ ")
            # printLog(dic)

            e_path = embed_path / f"iter_{epoch + 1}_embeds.npy"
            # l_path = embed_path / f"iter_{epoch + 1}_labels.npy"
            np.save(file=e_path.__str__(), arr=reprs)
            np.save(file=e_path.__str__(), arr=labels)
            if True:
                save_model(
                    args,
                    model,
                    optimizer,
                    optimizer_head,
                    epoch + 1,
                    path=checkpoint_path,
                    id=run.id,
                    best=best,
                )

        if (epoch + 1) % 5 == 0:
            save_model_10(
                args,
                model,
                optimizer,
                optimizer_head,
                epoch + 1,
                path=checkpoint_path,
                id=run.id,
            )
    return best_score


def XXXXXXXX():
    pass


if __name__ == "__main__":

    torch.cuda.empty_cache()
    parser = get_args_parser()
    args = parser.parse_args()
    # args.epochs = 50
    args.save_freq = 1
    # args.wb_mode = "offline"   #"offline"
    args.do_colab = False

    args.dataset_dir = "datasets" #if not args.do_colab else "stc-datasets"

    # args.resume = False
    # args.check_id = "95------20t5-eda=jzji5azk=t5_mixcse_mlm_full"
    args.init_eval = False
    do_multi_pos = False
    do_bank = True
    args.num_workers = 2
    # args.batch_size = 32# 256
    args.grad_steps = 1
    mode = args.wb_mode
    # global run_name, run_note
    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/
    if args.resume:
        print("resume On load to folder:")
        print(args.check_id)


    writer = SummaryWriter(log_dir="logs", filename_suffix="MIX")
    tb_writer = writer
    wb.tensorboard.patch(root_logdir="logs", pytorch=True)
    run = wb.init(
        project="t5_memory_bank",
        mode=args.wb_mode,
        group=args.dataset,
        resume="allow",
        # tensorboard=True,
        # sync_tensorboard=True,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MODEL_DIR"] = "../model"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # if not os.path.exists(args.model_path):
    #    os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # model and optimizer
    # text_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    # path = "perceptiveshawty/rankcse-listmle-bert-base-uncased"
    # path = "sosuke/ease-bert-base-uncased"
    path = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    text_model = AutoModel.from_pretrained(path)
    writer.add_text("wb_config", str(run.config))
    writer.add_text("arguments", str(args))
    class_num = args.class_num
    model = Network(
        backbone=text_model,
        args=args,
        feature_dim=args.feature_dim,
        class_num=class_num,
        config=config,
        tokenizer=tokenizer,
    )
    model = model.to("cuda")

    optimizer = torch.optim.SGD(
        model.backbone.parameters(),
        lr=args.lr_backbone,
        weight_decay=args.weight_decay,
    )
    optimizer_head = torch.optim.Adam(
        itertools.chain(
            model.instance_projector.parameters(),
            model.cluster_projector.parameters(),
        ),
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )
    run_note = "t5_bank_memory"
    s = "=".join([run.name, run.id, run_note])
    run_name = s
    (
        best_check_path,
        checkpoint_path,
        embed_path,
        last_check_path,
    ) = update_args(args, s)

    def upppppp():
        pass

    if args.resume == True:
        # model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        if True:  # or best_check_path.exists():
            # p = r"C:\Users\ComInSys\Desktop\vahidi-workspace\2022-IJCV-TCL-fork\tcl-official-repo\save\Stackoverflow\b3uga5x8----225-bert\checkpoints\last_10_epoch.tar"
            # best_p = r"save\Stackoverflow\=dei4h8ej=t5_2_51_100_init_82\last_10_epoch.tar"
            # best_p = r"save\Stackoverflow\=bzmgbtho=t5_mixcse_mlm_full\last_10_epoch.tar"
            # best_p = r"save\Stackoverflow\=p2c8ih5c=t5_mixcse_mlm_full\last_10_epoch.tar"
            # check_p = Path(best_p).resolve()
            check_p = last_check_path
            printLog("loading checkpoint...")
            checkpoint = torch.load(check_p.__str__())
            model.load_state_dict(checkpoint["net"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_head.load_state_dict(checkpoint["optimizer_head"])
            args.start_epoch = checkpoint["epoch"] + 1
            start_itr = args.start_epoch
            run.config.update(args, allow_val_change=True)
            printLog("resuming from epoch ", args.start_epoch)
        else:
            printLog("checkpoint path doesn't exist")
            printLog("training from scratch")
    else:
        printLog("training from scratch")

    # loss
    loss_device = torch.device("cuda")
    criterion = loss_git.ContrastiveLoss(
        args.batch_size,
        args.batch_size,
        class_num,
        args.instance_temperature,
        args.cluster_temperature,
        loss_device,
    ).to(loss_device)

    printLog(s)
    run.tags = [
        args.dataset.lower()[:5],
        s,
        # str(run.name),
        str(args.start_epoch),
        str(args.epochs),
        "multi_pos",
        "train",
    ]
    run.name = "-".join(run.tags) + run_note

    run.config.update(args, allow_val_change=True)
    (Path() / "logs").mkdir(exist_ok=True, parents=True)
    printLog(args)
    printLog("wandb run id: ", run.id)
    printLog(
        best_check_path.__str__(),
        checkpoint_path.__str__(),
        embed_path.__str__(),
        sep="\n",
    )

    sents, label = [], []
    with open(
        os.path.join(args.dataset_dir, args.dataset + ".txt"),
        "r",
        encoding="utf8",
    ) as f1:
        for line in f1:
            sents.append(line.strip("\n"))
    with open(
        os.path.join(args.dataset_dir, args.dataset + "_gnd.txt"),
        "r",
        encoding="utf8",
    ) as f2:
        for line in f2:
            label.append(line.strip("\n"))
    evaldataset = EvalDatasetIterater(sents, label)
    first_score = best_score = {"avg": 0, "acc": 0, "nmi": 0}

    if args.init_eval:
        printLog("### initial eval")
        labels, reprs, first_score = evaluation(
            dataset=evaldataset,
            model=model,
            device="cuda",
            epoch=-1,
            args=args,
            best_score=None,
        )
        # global metric_list
        metric_list.append(first_score)
        printLog("reprs.shape=========== ", reprs.shape)
        best_score = first_score.copy()
        e_path = embed_path / f"iter_{-1}_embeds.npy"
        # l_path = embed_path / f"iter_{-1}_labels.npy"
        np.save(file=e_path.__str__(), arr=reprs)
        np.save(file=e_path.__str__(), arr=labels)
    printLog("### start training...")

    wb.watch(model)
    t = time.time()
    printLog("from epoch {} to epoch {}".format(args.start_epoch, args.epochs))
    try:
        if do_bank is True:
            best_score = train_last10_all(best_score)
           # best_score = train_bank_all(best_score)
        elif do_multi_pos is True:
            best_score = train_mutli_all(best_score)
        else:
            best_score = train_all(best_score)
    except Exception as e:
        printLog(args)
        printLog(best_score)
        printLog(e)
        printLog(run.name)
        printLog(run.tags)
        printLog(run.id)
        mdf = pd.DataFrame(metric_list)
        ldf = pd.DataFrame(loss_list)
        bdf = pd.DataFrame(batch_loss_dicts)
        print(mdf)
        print(ldf)
        print(bdf)
        writer.close()
        run.finish()

        raise e
    finally:
        import pandas as pd

        mdf = pd.DataFrame(metric_list)
        ldf = pd.DataFrame(loss_list)
        bdf = pd.DataFrame(batch_loss_dicts)
        bdf.to_csv(f"logs/{s}_batch_loss_pandas.csv", sep="\t")
        mdf.to_csv(f"logs/{s}_metric_pandas.csv", sep="\t")
        ldf.to_csv(f"logs/{s}_loss_pandas.csv", sep="\t")

        # printLog(f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    dif = (time.time() - t) / 60
    printLog(f"training time: {dif:.2f} MIN")
    run.summary.update({"t(m)/training-total": dif})
    printLog("first scores:", first_score)
    printLog("best scores:", best_score)
    run.summary.update({f"first/{k}": v for k, v in first_score.items()})
    run.summary.update({f"top/{k}": v for k, v in best_score.items()})
    printLog(run.name)
    printLog(s)
    printLog(run.tags)
    printLog(run.id)
    printLog(str(checkpoint_path))
    writer.close()
    run.finish()
