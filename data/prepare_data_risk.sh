#!/bin/bash 

python src/extract_notes.py \
    --cohort_file cohort/splits_mextract.p \
    --mimic_dir ~/ \
    --out_name mextract_df.p \
    --window_size '1 day'


python src/prepare_hourly_for_original.py \
    --cohort_file cohort/splits_mextract.p \
    --hourly_file /scratch/jinghuil1/mimic_extract/all_hourly_data.h5 \
    --out_name mextract_hourly.p \
    --window_size 24
    