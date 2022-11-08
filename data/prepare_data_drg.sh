#!/bin/bash 


for DRG in ms apr; do 
    python src/extract_notes.py \
        --cohort_file cohort/splits_drg_${DRG}_raw.p \
        --mimic_dir ~/ \
        --out_name drg_${DRG}_df.p \
        --window_size '2 day' \
        -U
done


for DRG in ms apr; do 

    python src/prepare_hourly_for_drg.py \
        --cohort_file cohort/splits_drg_${DRG}.p \
        --hourly_file /scratch/jinghuil1/mimic_extract/all_hourly_data.h5 \
        --out_name drg_${DRG}_hourly.p \
        --window_size 48

done
