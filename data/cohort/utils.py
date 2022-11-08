
import pandas as pd 


def clean_apr(df):
    # remove duplicates and stays with more than one code 
    df = df.drop_duplicates()
    
    tmp = (df.groupby('HADM_ID').count()['DRG_CODE'] > 1)
    hadms = tmp[tmp].index
    
    df = df[~df.HADM_ID.isin(hadms)]
    
    return df

def encode_drg(df):
    codes = sorted(df['DRG_CODE'].unique())
    
    drg2idx = {d:i for i, d in enumerate(codes)}
        
    df['DRG'] = df['DRG_CODE'].apply(lambda x: drg2idx[x])
    
    print("The number of unique codes:", len(drg2idx))
    
    return df
    

def split_cohort_df_by_subj(cohort_df, ratios=(0.1, 0.1), random_state_val=1234, random_state_test=1443):
    """
    split by subj
    ratios: (val, test)
    """
    subj_ct = cohort_df.SUBJECT_ID.value_counts()
    subj_sg = pd.Series(subj_ct[subj_ct==1].sort_index().index)
    subj_ml = pd.Series(subj_ct[subj_ct >1].sort_index().index)
    
    val_ratio, test_ratio = ratios
    
    # split test
    test_subj = subj_sg.sample(frac=test_ratio, random_state=random_state_test).tolist() + \
                subj_ml.sample(frac=test_ratio, random_state=random_state_test).tolist()
    
    subj_sg, subj_ml = subj_sg[~subj_sg.isin(test_subj)], subj_ml[~subj_ml.isin(test_subj)]
    
    # split val
    val_ratio = val_ratio / (1-test_ratio)
    
    val_subj  = subj_sg.sample(frac=val_ratio, random_state=random_state_val).tolist() + \
                subj_ml.sample(frac=val_ratio, random_state=random_state_val).tolist()
    
    subj_sg, subj_ml = subj_sg[~subj_sg.isin(val_subj)], subj_ml[~subj_ml.isin(val_subj)]
    
    train_subj = subj_sg.tolist() + subj_ml.tolist()
    
    # split cohort df
    train_df = cohort_df[cohort_df['SUBJECT_ID'].isin(train_subj)]
    val_df = cohort_df[cohort_df['SUBJECT_ID'].isin(val_subj)]
    test_df = cohort_df[cohort_df['SUBJECT_ID'].isin(test_subj)]
    
    assert len(train_df) + len(val_df) + len(test_df) == len(cohort_df)
    
    return train_df, val_df, test_df


def split_drg_cohort(cohort_df):
    tr, val, te = split_cohort_df_by_subj(cohort_df, ratios=(0.05, 0.1))

    # print('Number of subjects: ', 
    #     tr.SUBJECT_ID.nunique(), val.SUBJECT_ID.nunique(), te.SUBJECT_ID.nunique())

    # print('Number of hadms: ', 
    #     tr.HADM_ID.nunique(), val.HADM_ID.nunique(), te.HADM_ID.nunique())

    # print('Number of hadms: ', len(tr), len(val), len(te))

    return {
        "train": tr,
        "val": val, 
        "test": te
    }


