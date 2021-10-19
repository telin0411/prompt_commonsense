import torch
import torch.nn as nn
import argparse
import os
import sys
# import apex.amp as amp
import torch.cuda.amp as amp
from torch.cuda.amp import GradScaler, autocast
from time import time
import pandas as pd
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from model import Transformer
from dataloader import ExDataset
from utils import str2bool, print_log, setup_logger, pred_entity, csv2list
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
Train w/ Val:
python3 main.py --mode train \
--expt_dir results_log/semeval20 --expt_name roberta_reasons \
--model roberta-base --data_dir ./datasets/semeval_2020_task4 \
--run_name demo --lr 1e-6 --batch_size 64

Test:
"""


def main():
    parser = argparse.ArgumentParser(description=' Expt')

    # Experiment params
    parser.add_argument('--mode', type=str, help='train or test mode', choices=['train', 'test'], default='train')
    parser.add_argument('--expt_dir', type=str, help='root directory to save model & summaries', default='./exp')
    parser.add_argument('--expt_name', type=str, help='expt_dir/expt_name: organize experiments', default='ex')
    parser.add_argument('--run_name', type=str, help='expt_dir/expt_name/run_name: organize training runs', default='01')

    # Model params
    parser.add_argument('--model', type=str, help='transformer model (e.g. roberta-base)', default='roberta-large')
    parser.add_argument('--seq_len', type=int, help='tokenized input sequence length', default=128)
    parser.add_argument('--num_layers', type=int,
                        help='Number of hidden layers in transformers (default number if not provided)', default=-1)
    parser.add_argument('--num_cls', type=int, help='model num of classes', default=2)
    parser.add_argument('--ckpt', type=str, help='path to model checkpoint .pth file')
    parser.add_argument('--pretrained', type=str2bool, help='use pretrained encoder', default='true')

    # Data params
    parser.add_argument('--train_path', type=str, help='path to dataset file (json/tsv)', default='./datasets/sem-eval/sem-eval_train.json')
    parser.add_argument('--dev_path', type=str, help='path to dataset file (json/tsv)', default='./datasets/sem-eval/sem-eval_dev.json')
    parser.add_argument('--test_path', type=str, help='path to dataset file (json/tsv)', default='./datasets/sem-eval/sem-eval_test.json')
    parser.add_argument('--pred_file', type=str, help='prediction csv file, for "test" mode')

    # Training params
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--use_amp', type=str2bool, help='Automatic-Mixed Precision (T/F)', default='T')
    parser.add_argument('--acc_step', type=int, help='gradient accumulation steps', default=1)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--log_interval', type=int, help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval', type=int, help='save model after `n` weight update steps', default=10000)
    parser.add_argument('--val_size', type=int, help='validation set size for evaluating metrics', default=2048)
    parser.add_argument('--use_reason', type=str2bool, help='Using reasons (T/F)', default='T')

    # GPU params
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs (0,1,2,..) else -1', default="0")
    parser.add_argument('-cpu', help='use cpu only (for test)', action='store_true')
    parser.add_argument('--opt_lvl', type=int, help='Automatic-Mixed Precision: opt-level (O_)', default=1,
                        choices=[0, 1, 2, 3])

    # Misc params
    parser.add_argument('--num_workers', type=int, help='number of worker threads for Dataloader', default=1)
    parser.add_argument('-data_parallel',
                        help='Whether to use nn.dataparallel (currently available for BERT-based models)',
                        action='store_true')

    args = parser.parse_args()

    data = ExDataset("./datasets/sem-eval/sem-eval_dev.json", 'train', 'roberta-large')

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

    assert not (text2text and (args.use_amp) == 'T'), 'use_amp should be F when using T5-based models.'

    # Train params
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    accumulation_steps = args.acc_step

    # Train
    if args.mode == 'train':
        # Setup train log directory
        log_dir = os.path.join(args.expt_dir, args.expt_name, args.run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/
        writer = SummaryWriter(log_dir)

        # Train log file
        log_file = setup_logger(parser, log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        # Dataset & Dataloader
        train_dataset = ExDataset(args.train_path, args.mode, tokenizer=args.model, max_seq_len=args.seq_len)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        val_dataset = ExDataset(args.dev_path, 'test', tokenizer=args.model, max_seq_len=args.seq_len)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        tokenizer = train_dataset.get_tokenizer()

        # Split sizes
        train_size = train_dataset.__len__()
        val_size = val_dataset.__len__()
        log_msg = 'Train Data Size: {}\n'.format(train_size)
        log_msg += 'Validation Data Size: {}\n\n'.format(val_size)

        # Min of the total & subset size
        val_used_size = min(val_size, args.val_size)
        log_msg += 'Validation Accuracy is computed using {} samples. See --val_size\n'.format(val_used_size)

        log_msg += 'Total Number of Classes {}\n'.format(args.num_cls)
        print_log(log_msg, log_file)

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
        model.train()

        # Loss & Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scaler = GradScaler(enabled=args.use_amp)

        # model, optimizer = amp.initialize(model, optimizer, opt_level="O{}".format(args.opt_lvl))

        # Step & Epoch
        start_epoch = 1
        curr_step = 1
        best_val_acc = 0.0

        # Load model checkpoint file (if specified)
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)

            # Load model & optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load other info
            curr_step = checkpoint['curr_step']
            start_epoch = checkpoint['epoch']
            prev_loss = checkpoint['loss']

            log_msg = 'Resuming Training...\n'
            log_msg += 'Model successfully loaded from {}\n'.format(args.ckpt)
            log_msg += 'Training loss: {:2f} (from ckpt)\n'.format(prev_loss)

            print_log(log_msg, log_file)

        steps_per_epoch = len(train_loader)
        start_time = time()

        for epoch in range(start_epoch, start_epoch + n_epochs):
            for batch in tqdm(train_loader):
                # Load batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast(args.use_amp):
                    if text2text:
                        # Forward + Loss
                        output = model(batch)
                        loss = output[0]

                    else:
                        # Forward Pass
                        label_logits = model(batch)
                        label_gt = batch['label']

                        # Compute Loss
                        loss = criterion(label_logits, label_gt)

                if args.data_parallel:
                    loss = loss.mean()
                # Backward Pass
                loss /= accumulation_steps
                scaler.scale(loss).backward()

                if curr_step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Print Results - Loss value & Validation Accuracy
                if curr_step % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    if val_dataset:
                        val_metrics = pred_entity(model, val_loader, device, tokenizer)

                        # Reset the mode to training
                        model.train()

                        log_msg = 'Validation Accuracy: {:.2f} %'.format(val_metrics['accuracy'])

                        print_log(log_msg, log_file)

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                        epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed, time_left)

                    print_log(log_msg, log_file)

                # Save the model
                if curr_step % args.save_interval == 0:
                    path = os.path.join(log_dir, 'model_' + str(curr_step) + '.pth')

                    state_dict = {'model_state_dict': model.state_dict(),
                                  # 'optimizer_state_dict': optimizer.state_dict(),
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg = 'Saving the model at the {} step to directory:{}'.format(curr_step, log_dir)
                    print_log(log_msg, log_file)

                curr_step += 1

            # Validation accuracy on the entire set
            if val_dataset:
                total_val_size = val_dataset.__len__()
                val_metrics = pred_entity(model, val_loader, device, tokenizer)
                log_msg = '\nAfter {} epoch:\n'.format(epoch)
                log_msg += 'Validation Accuracy: {:.2f} % \n'.format(
                    val_metrics['accuracy'])

                print_log(log_msg, log_file)

                # Save best model after every epoch
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]

                    filename = 'ep_{}_{:.2f}k_acc_{:.4f}_{}.pth'.format(
                        epoch, curr_step / 1000, best_val_acc, args.model.replace('-', '_'))

                    path = os.path.join(log_dir, filename)

                    state_dict = {'model_state_dict': model.state_dict(),
                                  # 'optimizer_state_dict': optimizer.state_dict(),
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg = "** Best Performing Model: {:.2f} ** \nSaving weights at {}\n".format(best_val_acc, path)
                    print_log(log_msg, log_file)

                # Reset the mode to training
                model.train()

        writer.close()
        log_file.close()

    elif args.mode == 'test':
        # Dataloader
        test_dataset = ExDataset(args.test_path, args.mode, tokenizer=args.model, max_seq_len=args.seq_len)
        tokenizer = test_dataset.get_tokenizer()

        test_loader = DataLoader(test_dataset, args.batch_size, num_workers=args.num_workers, drop_last=False)

        # Model
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
        model.eval()

        # Load model weights
        map_location = device
        checkpoint = torch.load(args.ckpt, map_location=map_location)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Inference
        # test_metrics = compute_eval_metrics(model, test_loader, device, test_dataset.__len__())
        test_metrics = pred_entity(model, test_loader, device, tokenizer)

        print('Test Accuracy: {}'.format(test_metrics['accuracy']))


if __name__ == '__main__':
    main()
