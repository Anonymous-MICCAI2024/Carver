import os
import argparse

import numpy as np
from trainer import SemanticSeg
import pandas as pd
import random

from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, PATH_LIST, FOLD_NUM,TEST_PATH

import time


def get_cross_validation_by_kbr_sample(path_list, fold_num, current_fold):

    sid_csv_path = '/staff/ydli/projects/OReX/Data/UNet/origin_sid_map.csv'
    df = pd.read_csv(sid_csv_path)

    path_str = ''
    for str in path_list[0].split('/')[:-1]:
        path_str = path_str+'/'+str
    print('file path:', path_str)
    # print(path_list)
    ORIGIN_ID = '3042'
    origin_sample_list = []

    for case in path_list:
        if ORIGIN_ID in os.path.basename(case).split('.')[0]:
            origin_sample_list.append(os.path.basename(case).split('.')[0][:-3])

    # print(origin_sample_list)
    origin_sample_list = np.unique(origin_sample_list)
    # print(sample_list)
    origin_sample_list.sort()
    print('number of sample:',len(origin_sample_list))
    _len_ = len(origin_sample_list) // fold_num

    origin_train_id = []
    origin_validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        origin_validation_id.extend(origin_sample_list[start_index:])
        origin_train_id.extend(origin_sample_list[:start_index])
    else:
        origin_validation_id.extend(origin_sample_list[start_index:end_index])
        origin_train_id.extend(origin_sample_list[:start_index])
        origin_train_id.extend(origin_sample_list[end_index:])

    print('the ratio of origin train id & val id ', len(origin_train_id)/len(origin_validation_id))


    train_id = []
    validation_id = []
    for sid, origin_sid in zip(df['sid'], df['origin_sid']):
        if origin_sid in origin_train_id:
            train_id.append(sid)
        else:
            validation_id.append(sid)

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('.')[0][:-3] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          ", Val set length ", len(validation_path))
    print('the ratio of train set & val set ', len(train_path)/len(validation_path))
    return train_path, validation_path


def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    # print(path_list)
    sample_list = list(set([os.path.basename(case).split('.')[0] for case in path_list]))
    # print(sample_list)
    sample_list.sort()
    print('number of sample:',len(sample_list))
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('.')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          "Val set length", len(validation_path))
    return train_path, validation_path



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train-cross',
                        choices=["train", 'train-cross', "inf"],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train-cross':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
    path_list = PATH_LIST
    # Training
    ###############################################
    if args.mode == 'train-cross':
        for current_fold in range(3, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            train_path, val_path = get_cross_validation_by_kbr_sample(path_list, FOLD_NUM, current_fold)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))
            break


    if args.mode == 'train':
        print("=== Training Fold ", CURRENT_FOLD, " ===")
        train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM,CURRENT_FOLD)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    elif args.mode == 'inf':
        test_path = TEST_PATH
        current_fold = 5
        train_path, val_path = get_cross_validation_by_kbr_sample(test_path, FOLD_NUM, current_fold)
        print('test len: %d'%len(val_path))
        start_time = time.time()
        result = segnetwork.inference(val_path)
        print('run time:%.4f' % (time.time() - start_time))
        print('ave dice:%.4f' % (result))
    ###############################################
