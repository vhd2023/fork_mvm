# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from EDA import *

# arguments to be parsed from command line
import argparse

from EDA.eda import eda

# generate more data with standard augmentation
def gen_eda(sents, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    # writer = open(output_file, 'w', encoding="utf8")
    # lines = open(train_orig, 'r', encoding="utf8").readlines()
    augs = []
    for i, sent in enumerate(sents):
        # parts = line[:-1].split('\t')
        # label = parts[0]
        aug_sentence = eda(sent, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        augs.append(aug_sentence[0])
    #     for aug_sentence in aug_sentences:
    #         writer.write(aug_sentence + '\n')
    return augs
    # writer.close()
    # print("generated augmented sentences with EDA for " + train_orig + " to " + output_file + " with num_aug=" + str(
    #     num_aug))
