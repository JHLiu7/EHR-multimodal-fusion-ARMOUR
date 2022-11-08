
import numpy as np 
import pandas as pd 

import re, os

ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']

def simple_imputer(df):
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]].copy()
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(0)
    
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.groupby(ID_COLS).cumsum() # a fix here
    hours_of_absence.columns.set_names(['LEVEL2', 'Aggregation Function'], inplace=True)
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].groupby(ID_COLS).fillna(method='ffill') #.fillna(100)
    
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out


def load_cohort(cohort_path):
    cohort = pd.read_pickle(cohort_path)

    if type(cohort) == dict:
        info_tr, info_dev, info_te = [cohort[s] for s in ['train', 'val', 'test']]

    elif type(cohort) == list:
        info_tr, info_dev, info_te = cohort 

    train_stay, dev_stay, test_stay = [df['ICUSTAY_ID'].values.tolist() for df in (info_tr, info_dev, info_te)]

    return [info_tr, info_dev, info_te], [train_stay, dev_stay, test_stay]

def load_noteevents(mimic_dir):

    note_col = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'DESCRIPTION', 'TEXT']
    noteevents_df = pd.read_csv(os.path.join(mimic_dir, 'NOTEEVENTS.csv.gz'), usecols=note_col, dtype={'CHARTTIME':object})
    noteevents_df.dropna(subset=['HADM_ID'], inplace=True)
    noteevents_df['HADM_ID'] = noteevents_df['HADM_ID'].astype(int)

    return noteevents_df

def _strip_phi(t):
    t = re.sub(r'\[\*\*.*?\*\*\]', ' ', t)
    t = re.sub(r'_', ' ', t)
    t = re.sub(r"`", '', t)
    t = re.sub(r"''", '', t)
    t = re.sub(r'"', '', t)
    return t


def unnest_df(value_df, hour=48):
    IDS = pd.Series([(a,b,c) for a,b,c,d in value_df.index.tolist()]).unique()
    ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
    
    icu_df = pd.DataFrame.from_records(IDS, columns=ID_COLS)
    icu_df['tmp']=10
    return unnest_visit(icu_df, value_df, hour=hour)

def unnest_visit(icu_df, value_df, hour=24):
    ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
    icu_df=icu_df.set_index('ICUSTAY_ID'.lower())
    
    missing_hours_fill =range_unnest_hour(icu_df, 'tmp', hour=hour, out_col_name='hours_in')
    missing_hours_fill['tmp'] = np.NaN
    
    fill_df = icu_df.reset_index()[ID_COLS].join(missing_hours_fill, on='icustay_id')
    fill_df.set_index(ID_COLS+['hours_in'], inplace=True)
    
    idx=pd.IndexSlice
    new_df = value_df.reindex(fill_df.index).copy()
    new_df.loc[:, idx[:, 'count']] = new_df.loc[:, idx[:, 'count']].fillna(0)
    return new_df

def range_unnest_hour(df, col, hour=24, out_col_name='hours_in', reset_index=False):
    if out_col_name is None: out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(hour)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index: col_flat = col_flat.set_index(df.index.names[0])
    return col_flat
