import pandas as pd
import pickle as pk
import numpy as np
from tqdm import tqdm

import os, re, argparse, logging

from utils import load_cohort, load_noteevents, _strip_phi

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def get_cohort_notes_raw(cohort_df, note_df, input_time_window, min_note_len=5):

    hadms = cohort_df['HADM_ID'].astype(int).tolist()
    logging.info(f'Processing {len(hadms)} hadms..')

    note = note_df.copy()
    note =  note[note['HADM_ID'].isin(hadms)]
    logging.info(f'Found {len(note)} raw notes for the cohort')

    def _strip_len(x): return len(x.strip())
    note['length'] = note['TEXT'].apply(_strip_len)
    note = note[note['length']>0]
    logging.info(f'...removing empty notes: {len(note)}')

    # fix charttime
    note['CHARTTIME_fix'] = note.apply(lambda r: r['CHARTTIME'] if type(r['CHARTTIME']) is str else r['CHARTDATE'], axis=1)
    note['CHART_TIME'] = pd.to_datetime(note['CHARTTIME_fix'])

    # remove exact duplicates
    note = note.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'CHART_TIME', 'CATEGORY', 'TEXT'])
    logging.info(f'...removing exact duplicates (same content and metadata): {len(note)}')

    # remove discharge summaries
    note = note[note.CATEGORY != 'Discharge summary']
    logging.info(f'...dropping discharge summaries: {len(note)}')

    # filter by time and input window
    note = note.merge(cohort_df[['HADM_ID', 'INTIME']], on=['HADM_ID'])
    time_mask = note['CHART_TIME'] <= pd.to_datetime(note['INTIME']) + pd.Timedelta(input_time_window)
    note = note[time_mask]

    logging.info(f'{len(note)} notes filtered by input time threshold: {input_time_window}')


    # simple clean 
    tqdm.pandas()
    note['CLEAN_TEXT'] = note['TEXT'].progress_apply(_strip_phi)

    def _count_len(x): return len(x.strip().split())
    note['clength'] = note['CLEAN_TEXT'].apply(_count_len)
    note = note[note['clength']>=min_note_len]
    logging.info(f'...removing empty or super short notes: {len(note)}')

    # post process
    note = note.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHART_TIME', 'CATEGORY', 'length'])
    col_drop = ['CHARTDATE', 'CHARTTIME', 'CHARTTIME_fix', 'TEXT', 'INTIME', 'length', 'clength']
    ncol = note.columns.drop(col_drop)
    
    final_df = note[ncol]

    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_file", default='split.p', type=str, help='pickled file with defined train/dev/test')
    parser.add_argument("--mimic_dir", default='~/physionet.org/files/mimiciii/1.4', type=str, help="Dir for MIMIC-III")

    parser.add_argument("--output_dir", default='notes_raw', type=str, help='output dir')
    parser.add_argument("--out_name", default='notes.p', type=str, help='save file name')

    parser.add_argument("--window_size", default='1 day', type=str)


    parser.add_argument('--update_cohort', '-U', action="store_const", const=True, default=False)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cohort_df = pd.concat(load_cohort(args.cohort_file)[0])
    noteevents_df = load_noteevents(args.mimic_dir)
    logging.info(f"Loaded raw note events and cohort file")
    
    note_df = get_cohort_notes_raw(cohort_df, noteevents_df, args.window_size)
    note_hadm = note_df.HADM_ID
    logging.info(f"{note_hadm.nunique()}/{cohort_df.HADM_ID.nunique()} hadms have notes")


    path = os.path.join(args.output_dir, args.out_name)
    with open(path, 'wb') as outf:
        pk.dump(note_df, outf)

    logging.info(f"dumped data to {path}")




    if args.update_cohort:
        assert note_hadm.nunique() < len(cohort_df), 'No need to update'

        tr, val, te = load_cohort(args.cohort_file)[0]

        print('Number of hadms: ', len(tr), len(val), len(te))

        tr, val, te = [
            df[df.HADM_ID.isin(note_hadm)]
            for df in [tr, val, te]
        ]

        print('Updated number of hadms: ', len(tr), len(val), len(te))

        update_cohort = {
            "train": tr,
            "val": val, 
            "test": te
        }

        with open(args.cohort_file.replace('_raw.p', '.p'), 'wb') as outf:
            pk.dump(update_cohort, outf)

        logging.info(f"updated cohort by fixing hadms")
        




if __name__ == '__main__':
    main()

