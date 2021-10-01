import torch.nn as nn
import random
import argparse
import pandas as pd
from time import time
from model import Transformer
from dataloader import BaseDataset
from transformers import AutoTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='Commonsense Dataset Dev')

    # Experiment params
    parser.add_argument('--mode', type=str, help='train or test mode', required=True)
    parser.add_argument('--expt_dir', type=str, help='root directory to save model & summaries')
    parser.add_argument('--expt_name', type=str, help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name', type=str, help='expt_dir/expt_name/run_name: organize training runs')
    parser.add_argument('--train_file', type=str, default='test',
                        help='The file containing train data to train.')
    parser.add_argument('--dev_file', type=str, default='dev',
                        help='The file containing dev data to be evaluated during training.')
    parser.add_argument('--test_file', type=str, default='test',
                        help='The file containing test data to evaluate in test mode.')

    # Model params
    parser.add_argument('--model', type=str, help='transformer model (e.g. roberta-base)', required=True)
    parser.add_argument('--num_layers', type=int,
                        help='Number of hidden layers in transformers (default number if not provided)', default=-1)
    parser.add_argument('--seq_len', type=int, help='tokenized input sequence length', default=256)
    parser.add_argument('--num_cls', type=int, help='model number of classes', default=2)
    parser.add_argument('--ckpt', type=str, help='path to model checkpoint .pth file')

    # Data params
    parser.add_argument('--pred_file', type=str, help='address of prediction csv file, for "test" mode',
                        default='results.csv')
    parser.add_argument('--train_dataset', type=str, help='list of datasets seperated by commas', required=False)
    parser.add_argument('--dev_dataset', type=str, default=None, help='list of datasets seperated by commas', required=False)
    parser.add_argument('--test_dataset', type=str, default=None, help='list of datasets seperated by commas', required=False)

    # Training params
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--acc_step', type=int, help='gradient accumulation steps', default=1)
    parser.add_argument('--log_interval', type=int, help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval', type=int, help='save model after `n` weight update steps', default=30000)
    parser.add_argument('--val_size', type=int, help='validation set size for evaluating metrics', default=2048)
    parser.add_argument('--seed', type=int, help='validation set size for evaluating metrics', default=808)

    # GPU params
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs (0,1,2,..) seperated by comma', default='0')
    parser.add_argument('-data_parallel',
                        help='Whether to use nn.dataparallel (currently available for BERT-based models)',
                        action='store_true')
    parser.add_argument('--use_amp', type=str2bool, help='Automatic-Mixed Precision (T/F)', default='T')
    parser.add_argument('-cpu', help='use cpu only (for test)', action='store_true')

    # Misc params
    parser.add_argument('--num_workers', type=int, help='number of worker threads for Dataloader', default=1)

    # Parse Args
    args = parser.parse_args()

    # Random seed
    setup_seed(args.seed)

    # Multi-GPU
    device_ids = csv2list(args.gpu_ids, int)
    print('Selected GPUs: {}'.format(device_ids))

    # Device for loading dataset (batches)
    device = torch.device(device_ids[0])
    if args.cpu:
        device = torch.device('cpu')

    # Text-to-Text
    text2text = ('t5' in args.model)
    uniqa = ('unified' in args.model)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build Model
    model = Transformer(args.model, args.num_cls, text2text, device_ids, num_layers=args.num_layers)
    if args.data_parallel and not args.ckpt:
        model = nn.DataParallel(model, device_ids=device_ids)
        device = torch.device(f'cuda:{model.device_ids[0]}')

    if not model.parallelized:
        model.to(device)

    if type(model) != nn.DataParallel:
        if not model.parallelized:
            model.to(device)

    # if not model.parallelized:
    #     model.to(device)

    # Load model checkpoint file (if specified)
    if args.ckpt:
        # Load model & optimizer
        if "11b" in args.model:
            map_location = torch.device("cpu")
        else:
            map_location = device
        checkpoint = torch.load(args.ckpt, map_location=map_location)

        # Load model & optimizer
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.data_parallel:
            model = nn.DataParallel(model, device_ids=device_ids)
            device = torch.device(f'cuda:{model.device_ids[0]}')

        if not model.parallelized:
            model.to(device)

    model.eval()

    sent = None
    while sent != "exit()":
        sent = input("Sentence: ")
        batch = to_ids(sent, tokenizer, args)
        batch = {k: torch.Tensor(v).unsqueeze(0).to(device).long()
                 if k == "input_ids" else torch.Tensor(v).unsqueeze(0).to(device)
                 for k, v in batch.items()}
        if text2text:
            # Forward Pass (predict)
            if args.data_parallel:
                label_pred = model.module.generate(input_ids=batch['input_ids'],
                                                   attention_mask=batch['attention_mask'],
                                                   max_length=2)
            else:
                label_pred = model.generate(input_ids=batch['input_ids'],
                                            attention_mask=batch['attention_mask'],
                                            max_length=2)
            label_pred = [decode(x, tokenizer).strip() for x in label_pred]

        # Others
        else:
            # Forward Pass
            batch["tokens"] = batch["input_ids"]
            batch["attn_mask"] = batch["attention_mask"]
            label_logits = model(batch)
            label_pred = torch.argmax(label_logits, dim=1)

        print("Prediction: {}".format(label_pred))


def decode(token_ids, tokenizer):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def to_ids(sent, tokenizer, args):
    input_encoded = tokenizer.encode_plus(text=sent,
                                          add_special_tokens=False,
                                          padding='max_length',
                                          max_length=args.seq_len,
                                          truncation=True,
                                          return_attention_mask=True)
    return input_encoded


if __name__ == '__main__':
    main()
