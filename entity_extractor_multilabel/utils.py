import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


@torch.no_grad()
def pred_entity(model, dataloader, device, tokenizer):
    model.eval()
    input_decoded = []
    output_decoded = []
    label = []
    acc = []

    def decode(token_ids):
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    # Evaluate on mini-batches
    for batch in dataloader:
        batch = {k: v.to(device) if k != 'label_string' else v for k, v in batch.items()}

        # Forward Pass
        label_logits = model(batch)
        B, L = label_logits.shape
        pred_mask = torch.zeros(label_logits.shape, dtype=torch.int).to(device)
        pred_mask[label_logits > 0] = 1
        print(label_logits)

        input_decoded_batch = [decode(x) for x in batch['input_ids']]
        input_decoded += input_decoded_batch
        accuracy = (~ torch.logical_xor(batch['label_binary'], pred_mask)).sum() / (B * L)
        acc.append(accuracy.item())

        pred = batch['input_ids'] * pred_mask

        output_decoded_batch = [decode(x) for x in pred]
        output_decoded += output_decoded_batch
        label += [batch['label_string']]

        print(input_decoded_batch)
        print(output_decoded_batch)
        print(label)
        print(acc)

    acc = torch.tensor(acc).mean()

    metric = {'accuracy': acc,
              'statement': input_decoded,
              'entity': output_decoded,
              'label': label}

    return metric

# ---------------------------------------------------------------------------
def setup_logger(parser, log_dir, file_name='train_log.txt'):
    """
    Generates log file and writes the executed python flags for the current run,
    along with the training log (printed to console). \n

    This is helpful in maintaining experiment logs (with arguments). \n

    While resuming training, the new output log is simply appended to the previously created train log file.

    :param parser: argument parser object
    :param log_dir: file path (to create)
    :param file_name: log file name
    :return: train log file
    """
    log_file_path = os.path.join(log_dir, file_name)

    log_file = open(log_file_path, 'a+')

    # python3 file_name.py
    log_file.write('python3 ' + sys.argv[0] + '\n')

    # Add all the arguments (key value)
    args = parser.parse_args()

    for key, value in vars(args).items():
        # write to train log file
        log_file.write('--' + key + ' ' + str(value) + '\n')

    log_file.write('\n\n')
    log_file.flush()

    return log_file


def print_log(msg, log_file):
    """
    :param str msg: Message to be printed & logged
    :param file log_file: log file
    """
    log_file.write(msg + '\n')
    log_file.flush()

    print(msg)


def csv2list(v, cast=str):
    assert type(v) == str, 'Converts: comma-separated string --> list of strings'
    return [cast(s.strip()) for s in v.split(',')]


def str2bool(v):
    v = v.lower()
    assert v in ['true', 'false', 't', 'f', '1', '0'], 'Option requires: "true" or "false"'
    return v in ['true', 't', '1']


def _shuffle(lst):
    np.random.seed(0)
    np.random.shuffle(lst)

    return lst


def train_val_split(data, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2):
    # Shuffle & Split data
    _shuffle(data)
    split_idx = int(len(data) * train_ratio)

    # train set
    data_train = data[:split_idx]

    # validation set
    rest = data[split_idx:]
    dev_split = int(dev_ratio * len(data))
    data_val = rest[:dev_split]
    rest = rest[dev_split:]
    test_split = int(test_ratio * len(data))
    data_test = rest[:test_split]
    return data_train, data_val, data_test
