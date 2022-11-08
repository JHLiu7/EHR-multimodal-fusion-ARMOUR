import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, auc, precision_recall_curve, accuracy_score
from collections import defaultdict



def _print_score_dicts(score_dicts, args):

    collected = defaultdict(list)
    for d in score_dicts:
        for k,v in d.items():
            collected[k].append(v)
    
    metrics, nums = [], []
    for metric, scores in collected.items():
        metrics.append(metric.upper())
        nums.append( '{:.3f} ({:.4f})'.format(np.mean(scores), np.std(scores)) )

    ## print
    print(f'\nSpreaded scores for {args.task} with {args.modality} on {len(score_dicts)} runs')
    for k, v in collected.items():
        print(k)
        print([f'{i:.5f}' for i in v])
    print()

    title = f'\nAggregated scores on {len(score_dicts)} runs'
    l_metric, l_num = '\t'.join(metrics), '\t'.join(nums)

    print(title)
    print(l_metric)
    print(l_num)
    print()

    main_num = l_num.split('\t')[0].split(' ')[0]
    return main_num


def evaluate_predict_output(output, task):
    
    y_hats, ys, stays = parse_predict_output(output)

    scorer = Evaluator(task)

    scores, _ = scorer.eval_all_scores(y_hats, ys, print_out=True)

    main_score = scores[scorer.score_main]

    return main_score, scores



def parse_predict_output(output, numpy=False):

    y_hats, ys, stays = [], [], []

    for t in output:
        y_hats.append(t[0].detach().cpu())
        ys.append(t[1].detach().cpu())
        stays.append(t[2])

    if numpy:
        y_hats= np.concatenate(y_hats)
        ys = np.concatenate(ys)
    else:
        y_hats = torch.cat(y_hats)
        ys = torch.cat(ys)

    stays = np.concatenate(stays)
    
    return y_hats, ys, stays




class Evaluator:
    def __init__(self, task):

        self.task = task 

        if ('mort' in task) or ('los' in task):
            self.score_func = _binary_scores
            self.score_main = 'auroc'
        else:
            self.score_func = _multiclass_scores
            self.score_main = 'acc'


    def eval_all_scores(self, logits, y, print_out=False):

        y_pred = self.get_ypred_from_logits(logits).numpy()
        y = y.numpy()

        scores, line = self.score_func(y_pred=y_pred, y=y)

        if print_out:
            print('Eval results')
            print(line)
            print()

        return scores, line


    def get_ypred_from_logits(self, logits):

        if self.score_main == 'acc':
            y_pred = torch.argmax(logits, -1)
        else:
            assert logits.size(-1) == 2
            y_pred = torch.softmax(logits, -1)[:, -1]
        return y_pred

    def get_main_score(self, logits, y):
        scores, _ = self.eval_all_scores(logits, y)
        return scores[self.score_main]


    


def _scores2str(scores):
    # slist = [f'{s*100:.1f}' for s in scores]
    slist = [f'{s:.3f}' for s in scores]
    
    line = '\t'.join(slist)
    return line


def _binary_scores(y_pred, y):
    # y_pred: 0~1, softmax applied
    auroc = roc_auc_score(y, y_pred)
    prec, rec, _ = precision_recall_curve(y, y_pred)
    aupr = auc(rec, prec)
    y_flat = np.around(y_pred, 0).astype(int)
    f1 = f1_score(y, y_flat)
    acc = accuracy_score(y, y_flat)

    out = {
        'auroc': auroc, 'aupr': aupr, 'f1': f1, 'acc': acc
    }

    line1 = f'AUCROC\tAUCPR\tF1\tACC\n'
    line2 = _scores2str([auroc, aupr, f1, acc])
    line = line1 + line2

    return out, line

def _multiclass_scores(y_pred, y):
    # y_pred: int
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro', labels=np.unique(y))
    f1_micro = f1_score(y, y_pred, average='micro', labels=np.unique(y))

    out = {
        'acc': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro
    }

    line1 = f'Acc\tMacroF1\tMicroF1\n'
    line2 = _scores2str([acc, f1_macro, f1_micro])
    line = line1 + line2

    return out, line