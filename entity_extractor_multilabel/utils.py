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

    def decode(token_ids):
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    # Evaluate on mini-batches
    for batch in dataloader:
        batch = {k: v.to(device) if k != 'label_string' else v for k, v in batch.items()}

        # Forward Pass
        label_logits = model(batch)
        pred_mask = torch.zeros(label_logits.shape, dtype=torch.int).to(device)
        pred_mask[label_logits > 0.5] = 1
        # print(pred_mask)

        input_decoded += [decode(x) for x in batch['input_ids']]

        pred = batch['input_ids'] * pred_mask

        output_decoded += [decode(x) for x in pred]
        label += batch['label_string']

    acc = compute_acc(output_decoded, label)
    print(acc)

    metric = {'accuracy': acc,
              'statement': input_decoded,
              'entity': output_decoded,
              'label': label}

    return metric

def compute_acc(source, target):

    print("===================source====================")
    print(source)
    print("===================target====================")
    print(target)

    assert len(source) % 2 == 0, "source need a factor of 2"
    acc = []
    for idx, _ in enumerate(source):
        words_source = source[idx].split()
        words_target = target[idx].split()
        cnt_correct = 0
        for word in words_source:
            if word in words_target:
                cnt_correct += 1

        acc.append(cnt_correct/len(words_target))

    return 100 * torch.tensor(acc, dtype=torch.float).mean()

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
