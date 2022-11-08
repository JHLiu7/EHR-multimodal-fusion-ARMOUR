import numpy as np, logging, os, pickle as pk
import torch

from data_utils import ExpDataModule
from eval_utils import Evaluator, evaluate_predict_output
from modules import ExpModule, init_trainer

from options import args 

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')

def _get_hp(hparam):
    hp = {
        'lr': hparam.lr,
        'wd': hparam.wd,
        'txt_size': hparam.txt_size,
        'ts_size': hparam.ts_size,
        'num_attention_heads': hparam.num_attention_heads,
        'intermediate_multiplier': hparam.intermediate_multiplier,
        'dropout': hparam.dropout,
        'dropout_grud': hparam.dropout_grud,
        'queue_size': hparam.queue_size,
        'contrast_embed_dim': hparam.contrast_embed_dim,
        'temperature': hparam.temp,
        'alpha': hparam.alpha,
    }

    return hp 

def main():

    dm = ExpDataModule(args)
    dm.setup()

    model = ExpModule.load_from_checkpoint(checkpoint_path=args.load_ckpt, strict=False)
    model.model.grud._init_x_mean(dm.X_mean)

    if torch.cuda.is_available():
        model.model.cuda()
    model.model.eval()

    if hasattr(model.model.fusion, 'train_stage'): model.model.fusion.train_stage=False 

    hp = _get_hp(model.hparams)
    logging.info(f'Loaded ckpt from {args.load_ckpt} with hparams:')
    print(str(hp))
    print()


    trainer = init_trainer(args)

    dloaders = {
        'train': dm.train_dataloader(),
        'val': dm.val_dataloader(),
        'test': dm.test_dataloader(),
    }

    outputs = {
        fold: trainer.predict(model, dataloaders=dloaders[fold])
        for fold in ['train', 'val', 'test']
    }

    dsizes = {
        fold: len(dset)
        for fold, dset in zip(['train', 'val', 'test'], [dm.train_dataset, dm.val_dataset, dm.test_dataset])
    }
        
    print()
    print()

    for fold in ['train', 'val', 'test']:

        print(f'Results on {fold.upper()} SET ({dsizes[fold]} cases)')

        _ = evaluate_predict_output(outputs[fold], args.task)

        print()


if __name__ == '__main__':
    main()
