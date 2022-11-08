
import numpy as np, logging
from collections import defaultdict

import pytorch_lightning as pl

from data_utils import ExpDataModule
from eval_utils import Evaluator, evaluate_predict_output
from modules import ExpModule, init_trainer

from options import args 

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')


def main_run(args, dm=None, seed=None):

    if seed is not None:
        pl.seed_everything(seed)
        logging.info(f'Set seed to {seed}')

    if dm is None:
        dm = ExpDataModule(args)
        dm.setup()

    model = ExpModule(args)

    if args.modality != 'text' and args.baseline_type != 'early':
        model.model.grud._init_x_mean(dm.X_mean)

    trainer = init_trainer(args)
    trainer.fit(model, dm)

    output = trainer.predict(model, dataloaders=dm.test_dataloader(), ckpt_path='best')
    main_score, scores = evaluate_predict_output(output, args.task)

    return main_score, scores

def seed_runs(args):

    score_dicts = []
    seeds = [3407, 34071, 34072, 340, 1234]
    if args.debug: seeds = seeds[:2]
    for seed in seeds:
        main_score, scores = main_run(args, seed=seed)
        score_dicts.append(scores)

    _ = _print_score_dicts(score_dicts, args)


if __name__ == '__main__':
    # main_run(args, seed=3407)
    seed_runs(args)

    