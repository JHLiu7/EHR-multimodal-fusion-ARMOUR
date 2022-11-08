import os, logging
import numpy as np 
import pandas as pd 
import pickle as pk 

import argparse

from tqdm import tqdm

from utils import load_cohort, simple_imputer, unnest_df

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_file", default='split.p', type=str, help='pickled file with defined train/dev/test')
    
    parser.add_argument("--hourly_file", default='all_hourly_data.h5', type=str, help='orginal file path')

    parser.add_argument("--output_dir", default='measurements', type=str, help='output dir')
    parser.add_argument("--out_name", default='input_104_hourly_df.p', type=str, help='save file name')

    parser.add_argument("--window_size", default=48, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load cohort
    cohort_dfs, cohort_stays = load_cohort(args.cohort_file)
    logging.info("cohort loaded in train/val/test")


    # load hourly data from original 
    logging.info("reading input data")
    df = pd.read_hdf(args.hourly_file, 'vitals_labs')


    # prepare
    logging.info("processing input data")
    df_train, df_dev, df_test = prepare_data_with_splits(df, args.window_size, *cohort_stays)

    # save 
    with open(os.path.join(args.output_dir, args.out_name), 'wb') as outf:
        pk.dump([df_train, df_dev, df_test], outf)

    logging.info("dumped data")


def prepare_data_with_splits(df, WINDOW_SIZE, train_stay, dev_stay, test_stay):
    '''
    stays: icustay_id
    raw_data: df loaded from raw file
    window_size: input hours

    return: df_train, df_dev, df_test (input for modeling)
    '''

    mask_hourly = df.index.get_level_values('hours_in') < WINDOW_SIZE
    df_train = df[(df.index.get_level_values('icustay_id').isin(train_stay)) & mask_hourly]
    df_dev = df[(df.index.get_level_values('icustay_id').isin(dev_stay)) & mask_hourly]
    df_test = df[(df.index.get_level_values('icustay_id').isin(test_stay)) & mask_hourly]


    logging.info("PREPROCESSING")
    # unnest data to align hours
    df_train = unnest_df(df_train, WINDOW_SIZE)
    df_dev = unnest_df(df_dev, WINDOW_SIZE)
    df_test = unnest_df(df_test, WINDOW_SIZE)


    # standardization
    logging.info("STANDARDIZATION")
    idx = pd.IndexSlice
    tr_means, tr_stds = df_train.loc[:, idx[:,'mean']].mean(axis=0), df_train.loc[:, idx[:,'mean']].std(axis=0)

    df_train.loc[:, idx[:,'mean']] = (df_train.loc[:, idx[:,'mean']] - tr_means)/tr_stds
    df_dev.loc[:, idx[:,'mean']] = (df_dev.loc[:, idx[:,'mean']] - tr_means)/tr_stds
    df_test.loc[:, idx[:,'mean']] = (df_test.loc[:, idx[:,'mean']] - tr_means)/tr_stds

    # imputation
    logging.info("IMPUTATION")
    # global_means = df_train.loc[:, idx[:, 'mean']].mean(axis=0)
    df_train, df_dev, df_test = [simple_imputer(df) for df in (df_train, df_dev, df_test)]
    logging.info("data ready")

    return df_train, df_dev, df_test


if __name__ == '__main__':
    main()
