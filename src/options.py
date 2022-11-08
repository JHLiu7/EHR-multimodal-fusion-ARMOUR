import argparse


def add_task_args(parent_parser):
    parser = parent_parser.add_argument_group("Task")

    parser.add_argument("--task", type=str, default="mort_hosp", choices=['mort_hosp', 'mort_icu', 'los_3', 'los_7', 'ms_drg', 'apr_drg'])

    parser.add_argument('--trim_cohort', '-T', action="store_const", const=True, default=False, help='remove cases w/ missing input modality')

    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--inf_dir", type=str, default="inf")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--modality", type=str, default="both")
    parser.add_argument("--note_encode_name", type=str, default="ClinicalBERT")
    parser.add_argument("--train_size_frac", type=float, default=1.)

    return parent_parser


def add_hp_args(parent_parser):
    parser = parent_parser.add_argument_group("Hyperparameter")

    parser.add_argument("--ts_size", type=int, default=128)
    parser.add_argument("--txt_size", type=int, default=768)

    parser.add_argument("--num_layer_ts", type=int, default=1)
    parser.add_argument("--num_layer_txt", type=int, default=1)
    parser.add_argument("--num_layer_cross", type=int, default=1)

    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--intermediate_multiplier", type=int, default=1)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dropout_grud", type=float, default=0.2)


    parser.add_argument('--add_contrast', '-C', action="store_const", const=True, default=False)

    parser.add_argument("--contrast_embed_dim", type=int, default=256)
    parser.add_argument("--queue_size", type=int, default=2000)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.2)

    return parent_parser



def add_train_args(parent_parser):
    parser = parent_parser.add_argument_group("Train")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--silent', action="store_const", const=True, default=False)
    parser.add_argument('--debug', '-D', action="store_const", const=True, default=False)

    parser.add_argument('--load_ckpt', "-L", default="", type=str)
    
    return parent_parser



parser = argparse.ArgumentParser()

parser = add_task_args(parser)
parser = add_hp_args(parser)
parser = add_train_args(parser)

args = parser.parse_args()
