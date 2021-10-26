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
    mask_token = tokenizer('<mask>', add_special_tokens=False)['input_ids'][0]
    becuz_token = tokenizer('because', add_special_tokens=False)['input_ids'][0]
    coma1_token = tokenizer(',', add_special_tokens=False)['input_ids'][0]
    coma2_token = tokenizer('.,', add_special_tokens=False)['input_ids'][0]

    acc = []

    def decode(token_ids):
        return tokenizer.decode(token_ids, skip_special_tokens=False)

    # Evaluate on mini-batches
    for batch in dataloader:
        batch = {k: v if 'str' in k else v.to(device) for k, v in batch.items()}

        # Forward Pass
        label_logits = model(batch)
        label_softmax = torch.nn.functional.softmax(label_logits, dim=1)

        one_hot = torch.nn.functional.one_hot(batch['input_ids'], label_logits.shape[1])
        one_hot = one_hot.sum(dim=1)
        one_hot[one_hot > 0] = 1

        # set special tokens = 0
        one_hot[:, 0:4] = 0
        one_hot[:, mask_token] = 0
        one_hot[:, becuz_token] = 0
        one_hot[:, coma1_token] = 0
        one_hot[:, coma2_token] = 0

        label_softmax = label_softmax * one_hot

        values, indices = torch.topk(label_softmax, 4, dim=1)

        # TODO: add some heuristic threshold from values to limit indices

        input_decoded += [decode(x) for x in batch['input_ids']]
        indices = indices.to('cpu')
        batch_pred_words = []
        for batch_token_id in indices:
            pred_words = []
            for token_id in batch_token_id:
                pred_words.append(decode(token_id))
            batch_pred_words.append(pred_words)

        acc.append(compute_precision(batch_pred_words, batch['label_string'], batch['input_string']))

    metric = {'accuracy': torch.tensor(acc).mean(),
              'statement': input_decoded,
              'entity': output_decoded,
              'label': label}

    return metric


def compute_acc(source, target, statement_b):
    """
    print("===================source====================")
    print(source)
    print("===================target====================")
    print(target)
    """
    # source [['b0_word_0',..., 'b0_word_k'],..., ['bn_word_0',..., 'bn_word_k']]
    # target [['b0_word_0',..., 'b0_word_k'],..., ['bn_word_0',..., 'bn_word_k']]
    acc = []
    for idx in range(len(source)):
        source_words = source[idx]
        target_words = target[idx].split()
        statement = statement_b[idx]
        for source_word in source_words:
            is_right = 0
            source_word = source_word.lower()
            for target_word in target_words:
                target_word = target_word.lower()
                source_word = source_word.replace(' ', '')
                source_word = string_match(source_word, statement)
                source_word = source_word.replace(' ', '')
                if source_word == target_word:
                    is_right = 1
                    break
            acc.append(is_right)
    return 100 * torch.tensor(acc, dtype=torch.float).mean()


def compute_precision(source, target, statement_b):
    """
    print("===================source====================")
    print(source)
    print("===================target====================")
    print(target)
    """
    # source [['b0_word_0',..., 'b0_word_k'],..., ['bn_word_0',..., 'bn_word_k']]
    # target [['b0_word_0',..., 'b0_word_k'],..., ['bn_word_0',..., 'bn_word_k']]
    acc = []
    for idx in range(len(source)):
        source_words = source[idx]
        target_words = target[idx].split()
        statement = statement_b[idx]
        for target_word in target_words:
            is_right = 0
            target_word = target_word.lower()
            for source_word in source_words:
                target_word = target_word.lower()
                source_word = source_word.replace(' ', '')
                source_word = string_match(source_word, statement)
                source_word = source_word.replace(' ', '')
                if source_word == target_word:
                    is_right = 1
                    break
            acc.append(is_right)
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


def string_match(sub_word: str, sentence: str):
    sub_word = sub_word.lower()
    sentence = sentence.lower()
    sentence = sentence.split()
    for word in sentence:
        if sub_word in word:
            return word
    return ''
