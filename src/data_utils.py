import numpy as np 
import pandas as pd 
from tqdm import tqdm

import torch, os

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

class ExpDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        # task/input config
        self.target = _task2target(args.task) # e.g., MORT_HOSP
        self.modality = args.modality # default to both
        self.batch_size = args.batch_size
        self.silent = args.silent

        # self.note_encode_dir = args.note_encode_dir
        self.note_encode_name = args.note_encode_name

        root_dir = os.path.abspath(args.root_dir)
        self.root_dir = root_dir

        # cohort df 
        cohort_name = _task2cohort(args.task)
        if not args.trim_cohort:
            all_cohort_df = pd.read_pickle(os.path.join(root_dir, 'data/cohort', f'splits_{cohort_name}.p'))
        else:
            all_cohort_df = pd.read_pickle(os.path.join(root_dir, 'data/cohort', f'trim_splits_{cohort_name}.p'))

        self.train_df, self.val_df, self.test_df = all_cohort_df['train'], all_cohort_df['val'], all_cohort_df['test']
        if args.debug:
            self.train_df, self.val_df, self.test_df = [df.head(200) for df in [self.train_df, self.val_df, self.test_df]]

        if args.train_size_frac < 1:
            # use frac of all train data
            num_sample = int(len(self.train_df) * args.train_size_frac)
            self.train_df = self.train_df.head(num_sample)


        if self.modality == 'both':
            self.my_collate_fn = collate_both

            self.data_txt = pd.read_pickle(os.path.join(root_dir, 'data/notes_encoded', f'{cohort_name}_df_{self.note_encode_name}.p' ))
            self.data_ts  = pd.read_pickle(os.path.join(root_dir, 'data/measurements', f'{cohort_name}_hourly.p'))
            self.X_mean = get_X_mean(self.data_ts[0])

        elif self.modality == 'struct':
            self.my_collate_fn = collate_ts

            self.data_ts  = pd.read_pickle(os.path.join(root_dir, 'data/measurements', f'{cohort_name}_hourly.p'))
            self.X_mean = get_X_mean(self.data_ts[0])

        elif self.modality == 'text':
            self.my_collate_fn = collate_txt

            self.data_txt = pd.read_pickle(os.path.join(root_dir, 'data/notes_encoded', f'{cohort_name}_df_{self.note_encode_name}.p' ))
    

    def init_datasets(self, cohort_df_list):

        if self.modality == 'struct':
            dataset_list = [
                StructDataset(self.target, cohort_df, data_df, self.silent)
                for cohort_df, data_df in zip(cohort_df_list, self.data_ts)
            ] 
        elif self.modality == 'text':
            dataset_list = [
                TextDataset(self.target, cohort_df, self.data_txt, silent=self.silent)
                for cohort_df in cohort_df_list
            ]
        elif self.modality == 'both':
            dataset_list = [
                BimodalDataset(self.target, cohort_df, [self.data_txt, data_df], silent=self.silent)
                for cohort_df, data_df in zip(cohort_df_list, self.data_ts)
            ]
        else:
            raise NotImplementedError

        return dataset_list


    def setup(self, stage=None):
        if stage == None:
            self.train_dataset, self.val_dataset, self.test_dataset = self.init_datasets(
                [self.train_df, self.val_df, self.test_df]
            )

        elif stage == 'test':
            # self.test_dataset = self.init_datasets([self.test_df])[0] # ignore this
            self.test_dataset = BimodalDataset(self.target, self.test_df, [self.data_txt, self.data_ts[-1]], silent=self.silent )


    def train_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.my_collate_fn, drop_last=True)
    def val_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate_fn)
    def test_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate_fn)



def _task2target(task):
    if task in ['ms_drg', 'apr_drg']:
        target = 'DRG'
    else:
        target = task.upper()
    return target

def _task2cohort(task):
    if 'ms' in task:
        cohort = 'drg_ms'
    elif 'apr' in task:
        cohort = 'drg_apr'
    else:
        cohort = 'mextract'
    return cohort
    


def collate_txt(batch):

    notes = []
    masks = []
    for b in batch:
        n = b[0]
        if len(n) > 0 and isinstance(n, list):
            notes.append(torch.tensor(np.array(n)))
            masks.append(torch.ones(len(n)).long())
        else:
            notes.append(torch.zeros(0, 768))
            masks.append(torch.zeros(1))

    # notes = [ torch.tensor(np.array(b[0])) for b in batch ]
    # masks = [ torch.ones(n.size(0)).long() for n in notes ]

    notes = pad_sequence(notes, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)

    labels = torch.tensor(np.array([ b[1] for b in batch ]))
    stays  = np.array([ b[2] for b in batch ])

    return notes, labels, masks, stays


def collate_ts(batch):

    input_window = batch[0][-1]

    series = []
    masks = []
    for b in batch:
        x = b[0]
        if x is not None:
            series.append(x)
            masks.append(np.ones(input_window))
        else:
            series.append(np.zeros((input_window, 312)))
            masks.append(np.zeros(input_window))

    series = torch.tensor(np.array(series))
    masks = torch.tensor(np.array(masks)).long()

    labels = torch.tensor(np.array([ b[1] for b in batch ]))
    stays  = np.array([ b[2] for b in batch ])

    return series, labels, masks, stays


def collate_both(batch):

    # ts 
    input_window = batch[0][-1]

    series = []
    masks_ts = []
    for b in batch:
        x = b[0][0]
        if x is not None:
            series.append(x)
            masks_ts.append(np.ones(input_window))
        else:
            series.append(np.zeros((input_window, 312)))
            masks_ts.append(np.zeros(input_window))

    series = torch.tensor(np.array(series))
    masks_ts = torch.tensor(np.array(masks_ts)).long()

    # txt
    notes = []
    masks_txt = []
    for b in batch:
        n = b[0][1]
        if len(n) > 0 and isinstance(n, list):
            notes.append(torch.tensor(np.array(n)))
            masks_txt.append(torch.ones(len(n)).long())
        else:
            notes.append(torch.zeros(0, 768))
            masks_txt.append(torch.zeros(1))

    notes = pad_sequence(notes, batch_first=True)
    masks_txt = pad_sequence(masks_txt, batch_first=True)

    # others
    labels = torch.tensor(np.array([ b[1] for b in batch ]))
    stays  = np.array([ b[2] for b in batch ])

    return (series, notes), labels, (masks_ts, masks_txt), stays


class TemplateDataset(Dataset):
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class TextDataset(TemplateDataset):
    def __init__(self, target, cohort_df, note_df, max_num_note=31, silent=False):
        super().__init__()

        self.data = []
        input_window = 48 if target == 'DRG' else 24

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):
            
            x, hadm = _query_note(row, note_df)
            x = x[::-1][:max_num_note] # keep latest note first 

            y = row[target]
            self.data.append([x, y, hadm, input_window])


class StructDataset(TemplateDataset):
    def __init__(self, target, cohort_df, ts_df, silent=False):
        super().__init__()

        self.data = []
        input_window = 48 if target == 'DRG' else 24

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):

            x, hadm = _query_ts(row, ts_df)

            y = row[target]
            self.data.append([x, y, hadm, input_window])



class BimodalDataset(TemplateDataset):
    def __init__(self, target, cohort_df, data_dfs, max_num_note=31, silent=False):
        super().__init__()

        assert len(data_dfs) == 2
        note_df, ts_df = data_dfs

        self.data = []
        input_window = 48 if target == 'DRG' else 24

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):

            x_ts, hadm = _query_ts(row, ts_df)
            x_txt, hadm2=_query_note(row, note_df)
            x_txt = x_txt[::-1][:max_num_note] # keep latest note first 

            assert hadm == hadm2 

            y = row[target]
            self.data.append([(x_ts, x_txt), y, hadm, input_window])



def _query_ts(row, ts_df):
    subj, hadm, icu = row['SUBJECT_ID'], row['HADM_ID'], row['ICUSTAY_ID']
    msk = id_msk(ts_df, subj, hadm, icu)

    if msk.any() == False:
        return None, hadm

    else:
        x = ts_df[msk].values
        return x, hadm 


def _query_note(row, note_df):
    hadm = row['HADM_ID']
    msk = note_df['HADM_ID'] == hadm

    if msk.any() == False:
        return [], hadm

    else:
        x = note_df[msk]['vector'].tolist()
        return x, hadm




def id_msk(df, subj, hadm, icu):
    msk1 = df.index.get_level_values('subject_id') == subj
    msk2 = df.index.get_level_values('hadm_id') == hadm
    msk3 = df.index.get_level_values('icustay_id') == icu
    return msk1 & msk2 & msk3

def to_3D_tensor(df):
    idx = pd.IndexSlice
    return np.dstack([df.loc[idx[:,:,:,i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))])

def get_X_mean(lvl2_train):
    X_mean = np.nanmean(
            to_3D_tensor(
                lvl2_train.loc[:, pd.IndexSlice[:, 'mean']] * 
                np.where((lvl2_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN)
            ),
            axis=0, keepdims=True
        ).transpose([0, 2, 1])
    X_mean = np.nan_to_num(X_mean,0)
    return X_mean


