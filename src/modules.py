import torch, os, random
import torch.nn as nn 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from layers import init_model
from eval_utils import Evaluator

class ExpModule(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters(args)

        self.model = init_model(self.hparams)

        self.scorer = Evaluator(self.hparams.task)

        self.batch_size = self.hparams.batch_size
        self.learning_rate = self.hparams.lr
        self.fusion = True if self.hparams.modality == 'both' else False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.wd)
        return optimizer

    def forward(self, input, target, mask):

        if isinstance(input, tuple):
            x1,x2 = input
            m1,m2 = mask
            logits, loss = self.model(x1,x2, target, m1,m2)
        else:
            logits, loss = self.model(input, target, mask)

        return logits, loss 

    def step(self, batch, train_stage=False):
        input, target, mask, stay = batch 

        logits, loss = self.forward(input, target, mask)

        return loss, (logits, target)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input, target, mask, stay = batch 

        if self.fusion:
            if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=False 
        logits, loss = self.forward(input, target, mask)

        return logits, target, stay

    def training_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        self.log('train_loss', loss, on_epoch=True, batch_size=self.batch_size)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        if self.fusion:
            if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=False 
        self.log('valid_loss', loss, batch_size=self.batch_size)
        return {'loss': loss, 'logits':logits, 'y':y}

    def test_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        if self.fusion:
            if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=False 
        self.log('test_loss', loss, on_epoch=True, batch_size=self.batch_size)
        return {'loss': loss, 'logits':logits, 'y':y}
    
    def validation_epoch_end(self, list_of_dict):
        all_logits, all_y = [], []
        for d in list_of_dict:
            all_logits.append(d['logits'])
            all_y.append(d['y'])

        logits = torch.cat(all_logits).detach().cpu()
        y = torch.cat(all_y).cpu()
        
        scores, line = self.scorer.eval_all_scores(logits, y)
        score = scores[self.scorer.score_main]
        self.log('valid_score', score)

        if not self.hparams.silent:
            print(line)

    
def init_trainer(args):

    RESULT_DIR = args.output_dir

    version = str(random.randint(0, 2000))
    log_dir = os.path.join(RESULT_DIR, 'log', args.modality)
    model_dir = os.path.join(RESULT_DIR, 'model', args.modality, args.task, version)

    device = None if args.device==-2 else [args.device] # device is int

    metric = 'valid_score'
    mode = 'max'

    logger = TensorBoardLogger(save_dir=log_dir, version=version, name=args.task)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, filename='{epoch}-{%s:.3f}' % metric, 
        monitor=metric, mode=mode, save_weights_only=True, 
    )
    early_stopping = EarlyStopping(
        monitor=metric, min_delta=0., patience=args.patience,
        verbose=False, mode=mode
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gpus=device,
        num_sanity_val_steps=0,
        max_epochs=args.epochs if not args.debug else 2, 
        enable_progress_bar=not args.silent,
    )
    return trainer