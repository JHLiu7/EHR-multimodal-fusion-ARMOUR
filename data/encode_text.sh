#!/bin/bash 

# gpu required

for NAME in drg_ms drg_apr mextract
do 

    for MODEL in emilyalsentzer/Bio_ClinicalBERT
    do

        python src/encode_notes.py \
            --note_df_name ${NAME}_df.p \
            --bert_model_path $MODEL 

    done
done 
