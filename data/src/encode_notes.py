import logging
import os

import torch
import datasets
import numpy as np
import pandas as pd
import pickle as pk

from tqdm import tqdm

import argparse 

import datasets

from transformers import AutoTokenizer, AutoModel


from torch.utils.data import DataLoader

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_encode_name(bert_model_path):

    toName = {
        "bert-base-uncased": "BERT",
        "emilyalsentzer/Bio_ClinicalBERT": "ClinicalBERT",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
        "jhliu/ClinicalNoteBERT-base-uncased-MIMIC-segment-note": "ClinicalNoteBERT",
    }

    return toName[bert_model_path]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--note_df_name", default='notes.p', type=str, help='save file name')
    parser.add_argument("--note_df_dir", default='notes_raw', type=str, help='input dir')

    parser.add_argument("--bert_model_path", default='', type=str)

    parser.add_argument("--bert_cache_dir", default='bert_cache_dir', type=str)

    parser.add_argument("--batch_size", default=64, type=int)

    parser.add_argument("--output_dir", default='notes_encoded', type=str, help='output dir')

    args = parser.parse_args()


    USE_CUDA = True


    inpath = os.path.join(args.note_df_dir, args.note_df_name)
    note_df = pd.read_pickle(inpath)
    # note_df = note_df.head(5000)
    logging.info(f"Loaded {len(note_df)} notes from {inpath}")


    model_path = args.bert_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.bert_cache_dir)
    model = AutoModel.from_pretrained(model_path, cache_dir=args.bert_cache_dir)
    model.eval()
    if USE_CUDA:
        model.cuda()


    # prepare data 
    notes = note_df['CLEAN_TEXT'].tolist()
    input_key = "text"
    raw_dataset = datasets.Dataset.from_dict({input_key: notes})

    def tokenize_function(examples):
        result = tokenizer(examples[input_key], padding=True, max_length=512, truncation=True)
        return result

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=[input_key]
    )
    tokenized_dataset.set_format(type="torch")


    # encode
    data_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size)
    num_batch = int(len(tokenized_dataset) / args.batch_size)

    cls_list = []
    for batch in tqdm(data_loader, total=num_batch):
        if USE_CUDA:
            batch = {k:v.cuda() for k,v in batch.items()}
        
        with torch.no_grad():
            out = model(**batch)

        cls = out.last_hidden_state[:, 0].cpu()

        cls_list.append(cls)

    all_cls = torch.cat(cls_list, 0)
    dim = all_cls.size(-1)


    # assign to df
    cls_ready = [i for i in all_cls.numpy()]
    assert len(notes) == len(cls_ready)

    note_df['vector'] = cls_ready
    note_df = note_df[note_df.columns.drop(['CLEAN_TEXT'])]


    # dump 
    bert_name = get_encode_name(args.bert_model_path)
    out_name = args.note_df_name.replace('.p', f'_{bert_name}.p')

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, out_name)
    with open(outpath, 'wb') as outf:
        pk.dump(note_df, outf)

    logging.info(f"Dumped data to {outpath}")


if __name__ == '__main__':
    main()
