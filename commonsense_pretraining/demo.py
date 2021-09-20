"""
Classifies sentences as T/F, given the input file (txt)
for the selected model (see ./model_ckpt/readme.md)

"""
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from model import Transformer
from utils import str2bool


"""
python3 demo.py \
--inp_txt ./datasets/ours/inp.txt \
--ckpt ./model_ckpt/ep_13_16k_acc_87.2984_roberta_large.pth \
--model roberta-large --gpu 0
"""


@torch.no_grad()
def run():
    parser = argparse.ArgumentParser(description='Local Demo')

    # I/O params
    parser.add_argument('--inp_txt',        type=str,       help='path to dataset file (text)', required=True)
    parser.add_argument('--pred_csv',       type=str,       help='prediction csv file; default: print to console')

    # Model params
    parser.add_argument('--model',          type=str,       help='transformer model (e.g. roberta-base)', required=True)
    parser.add_argument('--ckpt',           type=str,       help='path to model checkpoint .pth file', required=True)
    parser.add_argument('--seq_len',        type=int,       help='tokenized input sequence length', default=128)
    parser.add_argument('--num_cls',        type=int,       help='model num of classes', default=2)

    # Misc
    parser.add_argument('--gpu_ids',        type=int,       help='GPU ID; use -1 for CPU', nargs='+', default=0)
    parser.add_argument('--use_ram',        type=str2bool,  help='loads model to RAM for subsequent runs', default='false')
    # parser.add_argument('--batch_size',     type=int,       help='batch size', default=1)
    # parser.add_argument('--num_workers',    type=int,       help='number of worker threads for Dataloader', default=1)

    args = parser.parse_args()

    # Configs
    load_to_ram = args.use_ram
    gpu_id = args.gpu_ids[0]
    device = torch.device('cuda:{}'.format(gpu_id) if gpu_id != -1 and torch.cuda.is_available() else 'cpu')

    # Model
    model = Transformer(args.model, args.num_cls)
    model.to(device)

    # Checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if load_to_ram:
        print('** RAM mode **\n'
              'You can choose to re-compute, as you update the input file')

    # Evaluate
    while True:
        print('\nExecuting ...')
        # Dataset
        with open(args.inp_txt) as f:
            lines = f.read().strip().split('\n')

        outputs = []
        for line in tqdm(lines):
            # If input text
            if len(line) > 2 and not line.strip().startswith('#'):
                text, label = line.rsplit(',', 1)

                inp = _tokenize(text, args.model, args.seq_len)

                # Add dummy batch dim
                tokens = inp['tokens'].unsqueeze(dim=0).to(device)
                mask = inp['attn_mask'].unsqueeze(dim=0).to(device)

                # Forward pass
                logit = model(tokens, mask)
                _, pred_cls = logit.max(dim=-1)

                pred = pred_cls.item()
                label = int(label)

                is_correct = (pred == label)
                is_input = True
            else:
                text = line
                label, pred = '', ''
                is_correct = ''
                is_input = False

            outputs.append({'correct': is_correct,
                            'text': text,
                            'label_gt': label,
                            'label_pred': pred,
                            'is_input': is_input})
        # Output
        df = pd.DataFrame(outputs)

        # Accuracy
        df_inp = df[df['is_input']]
        correct = sum(df_inp['correct'].to_list())
        total = len(df_inp)
        accuracy = correct / total
        print('\nAccuracy: {:.2f} ({}/{})'.format(accuracy, correct, total))

        # Save as CSV
        if args.pred_csv:
            df = df[df['is_input']]
            df = df.drop(columns=['is_input'])

            df.to_csv(args.pred_csv, index=False)
            print('Saving as csv: {}'.format(args.pred_csv))

        # Print to console
        else:
            pprint_results(df)

        # Handle user input (in RAM mode)
        if load_to_ram:
            try_inp = True
            user_input = ''
            while try_inp:
                try:
                    user_input = str(input("Would you like to Compute (c) or Quit (q)? "))
                    try_inp = not (user_input.lower() in ['c', 'q'])
                except Exception as e:
                    print('Please type either "c" or "q"')
                    print(e)

            load_to_ram = (user_input == 'c')

        # Else terminate
        if not load_to_ram:
            print('\n -- Done! --')
            break


def _tokenize(text, tok_name, max_len):
    """
    Tokenizes raw text, pads to max length.

    :param str text: input sentence
    :param str tok_name: tokenizer name (model)
    :param int max_len: sequence length
    :return: tokens & attention mask
    """

    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    # Tokenized format: [CLS] [text] [PAD]
    tokens = [tokenizer.cls_token]
    tokens += tokenizer.tokenize(text)

    tokens = tokenizer.encode_plus(text=tokens,
                                   padding='max_length',
                                   max_length=max_len,
                                   add_special_tokens=False,
                                   return_attention_mask=True)

    token_ids = torch.tensor(tokens['input_ids'])
    attn_mask = torch.tensor(tokens['attention_mask'])

    # Output
    sample = {'tokens': token_ids,
              'attn_mask': attn_mask}
    return sample


def pprint_results(df_):
    """
    Pretty print results to console

    :param df_: data (input, prediction, ground-truth)
    :type df_: pd.DataFrame
    """
    # col_width = max(len(t) for t in texts) + 4    # padding
    print('\n' + 'Result' + '\t' + 'Answer' + '\t' + 'Predicted' + '\t' + 'Text')
    df_['correct'] = df_['correct'].apply(lambda x: 'Yes' if x else 'No')

    for row in df_.to_dict(orient='records'):
        # If input text
        if row['is_input']:
            print(row['correct'] + '\t' + str(row['label_gt']) + '\t' + str(row['label_pred']) + '\t' + row['text'])

        # Else - empty line or comment
        else:
            print('' + '\t' + row['text'])
            # sent.ljust(col_width) + label


if __name__ == '__main__':
    run()
