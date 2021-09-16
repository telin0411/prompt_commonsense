import argparse
from utils import *


def config():
    parser = argparse.ArgumentParser(description='Commonsense Dataset Dev')

    # Experiment params
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', required=True,
                        help='train or test mode')
    parser.add_argument('--expt_dir', type=str, default='./results',
                        help='root directory to save model & summaries')
    parser.add_argument('--expt_name', type=str, default='modelB',
                        help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name', type=str, default='dev_0915',
                        help='expt_dir/expt_name/run_name: organize training runs')
    parser.add_argument('--test_file', type=str, default='test',
                        help='The file containing test data to evaluate in test mode.')

    # Model params
    parser.add_argument('--model', type=str, default='roberta-base', required=True,
                        help='transformer model (e.g. roberta-base)')
    parser.add_argument('--model_name', type=str, default='model_a', choices=['model_a', 'model_b'],
                        help='model_a or model_b')
    parser.add_argument('--num_prompt_model_layer', type=int, default=-1,
                        help='Number of hidden layers in transformers (default number if not provided)')
    parser.add_argument('--num_task_model_layer', type=int, default=-1,
                        help='Number of hidden layers in transformers (default number if not provided)')
    parser.add_argument('--seq_len', type=int, default=64,
                        help='tokenized input sequence length')
    parser.add_argument('--num_cls', type=int, default=2,
                        help='model number of classes')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to model checkpoint .pth file')

    # Data params
    parser.add_argument('--pred_file', type=str, default='results.csv',
                        help='address of prediction csv file, for "test" mode')
    parser.add_argument('--data_dir', type=str, default='./datasets/com2sense', required=True,
                        help='The directory of data files, a directory that contains train.json, dev.json, test.json')
    parser.add_argument('--template', type=str, default='rather than',
                        help='template for transformation from causality to comparison')
    parser.add_argument('--mask_len', type=int, default=1,
                        help='the number of unknown mask')

    # Training params
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--acc_step', type=int, default=1,
                        help='gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='interval size for logging training summaries')
    parser.add_argument('--save_interval', type=int, default=30000,
                        help='save model after `n` weight update steps')
    parser.add_argument('--val_size', type=int, default=2048,
                        help='validation set size for evaluating metrics')
    parser.add_argument('--seed', type=int, default=430,
                        help='random seed for all random algorithm')

    # GPU params
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs (0,1,2,..) seperated by comma')
    parser.add_argument('--use_amp', type=str2bool, default='T',
                        help='Automatic-Mixed Precision (T/F)')
    parser.add_argument('-cpu', help='use cpu only (for test)', action='store_true')

    # Misc params
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of worker threads for Dataloader')

    # Parse Args
    return parser
