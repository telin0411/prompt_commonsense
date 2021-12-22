import os
import sys
import torch
import torch.multiprocessing
import torch.nn as nn
import csv
import argparse
import pandas as pd
from time import time
from model_gan import Transformer, Generator, Discriminator
from dataloader import GANPair
from append_pred_to_data import append_to_data

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_random_seed(random_seed: int):
    # set random seed for PyTorch and CUDA
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    # set random seed for Numpy
    # np.random.seed(random_seed)

    # set random seed for random
    # random.seed(random_seed)


def main():
    parser = argparse.ArgumentParser(description='Commonsense Dataset Dev')

    # Experiment params
    parser.add_argument('--mode', type=str, help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir', type=str, help='root directory to save model & summaries')
    parser.add_argument('--expt_name', type=str, help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name', type=str, help='expt_dir/expt_name/run_name: organize training runs')

    # Model params
    parser.add_argument('--model', type=str, help='transformer model (e.g. roberta-base)', required=True)
    parser.add_argument('--num_layers', type=int,
                        help='Number of hidden layers in transformers (default number if not provided)', default=-1)
    parser.add_argument('--seq_len', type=int, help='tokenized input sequence length', default=256)
    parser.add_argument('--num_cls', type=int, help='model number of classes', default=2)
    parser.add_argument('--ckpt', type=str, help='path to model checkpoint .pth file')
    parser.add_argument('--weights_path', type=str, help='the bart weights to generate embeddings')

    # Data params
    parser.add_argument('--pred_file', type=str, default='results.csv')
    parser.add_argument('--append_test_file', type=str, default='results_append.json')
    parser.add_argument('--test_file', type=str, default='test')

    parser.add_argument('--real_train_file', type=str, required=True)
    parser.add_argument('--fake_train_file', type=str, required=True)
    parser.add_argument('--real_valid_file', type=str)
    parser.add_argument('--fake_valid_file', type=str)

    # Training params
    parser.add_argument('--seed', type=int, help='random seed', default=888)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--acc_step', type=int, help='gradient accumulation steps', default=1)
    parser.add_argument('--log_interval', type=int, help='interval size for logging training summaries', default=3000)
    parser.add_argument('--save_interval', type=int, help='save model after `n` weight update steps', default=3000)
    parser.add_argument('--val_size', type=int, help='validation set size for evaluating metrics', default=2048)

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

    set_random_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system') 

    # Multi-GPU
    device_ids = csv2list(args.gpu_ids, int)
    print('Selected GPUs: {}'.format(device_ids))

    # Device for loading dataset (batches)
    #device = torch.device(device_ids[0])
    #if args.cpu:
    device = torch.device('cpu')

    # Text-to-Text
    text2text = 't5' in args.model or 'T0' in args.model or 'bart' in args.model

    assert not (text2text and args.use_amp == 'T'), 'use_amp should be F when using T5-based models.'
    # Train params
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    accumulation_steps = args.acc_step

    # Train
    if args.mode == 'train':
        # Ensure CUDA available for training
        assert torch.cuda.is_available(), 'No CUDA device for training!'

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
        train_datasets = GANPair(real_file=args.real_train_file, fake_file=args.fake_train_file,
                                 tokenizer=args.model, input_seq_len=args.seq_len)
        valid_datasets = GANPair(real_file=args.real_valid_file, fake_file=args.fake_valid_file,
                                 tokenizer=args.model, input_seq_len=args.seq_len)
        train_loader = DataLoader(train_datasets, batch_size, shuffle=True,
                                  drop_last=True, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_datasets, batch_size, shuffle=True,
                                       drop_last=True, num_workers=args.num_workers)


        # Tokenizer
        tokenizer = train_datasets.get_tokenizer()

        # Split sizes
        train_size = train_datasets.__len__()
        val_size = valid_datasets.__len__()
        log_msg = 'Train: {} \nValidation: {}\n\n'.format(train_size, val_size)

        # Min of the total & subset size
        val_used_size = min(val_size, args.val_size)
        log_msg += 'Validation Accuracy is computed using {} samples. See --val_size\n'.format(val_used_size)

        # log_msg += 'No. of Classes: {}\n'.format(args.num_cls)
        print_log(log_msg, log_file)

        # Build Model
        netG = Generator(args.model)
        netD = Discriminator("roberta-large")
        netG_real = Generator(args.model)

        netG.train()
        netD.train()
        netG_real.eval()

        # Load pre-trained BART weights
        pretrain_weights = torch.load(args.weights_path, map_location=device)
        netG_real.load_state_dict(pretrain_weights['model_state_dict'], strict=False)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizerG = torch.optim.Adam(netG.parameters(), lr)
        optimizerD = torch.optim.Adam(netD.parameters(), lr)
        optimizerG.zero_grad()
        optimizerD.zero_grad()

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # Load model checkpoint file (if specified)
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)

            # Load model & optimizer
            netG.load_state_dict(checkpoint['G']['model_state_dict'])
            netD.load_state_dict(checkpoint['D']['model_state_dict'])

            netG.to(device)
            netD.to(device)

            log_msg = 'Resuming Training...\n'
            log_msg += 'Model successfully loaded from {}\n'.format(args.ckpt)

            print_log(log_msg, log_file)

        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(args.epochs):
            i = -1
            # For each batch in the dataloader
            for data in tqdm(train_loader):
                i += 1
                real_data = data['real']
                fake_data = data['fake']
                real_data = {k: v.to(device) for k, v in real_data.items()}
                fake_data = {k: v.to(device) for k, v in fake_data.items()}

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                b_size = real_data['input_token_ids'].shape[0]
                label = torch.full((b_size,), real_label, dtype=torch.long, device=device)
                # Get real embedding by G
                with torch.no_grad():
                    real_embeddings = netG_real(real_data)
                # Forward pass real embeddings through D
                output = netD(real_embeddings)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate fake embeddings
                fake_embeddings = netG(fake_data)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake_embeddings)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake_embeddings)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, args.epochs, i, len(train_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # validation


                iters += 1

        writer.close()
        log_file.close()
    """
    elif args.mode == 'test':

        # Dataloader
        test_dataset = BartTransfer(file_path=args.test_file, tokenizer=args.model, input_seq_len=args.seq_len)

        loader = DataLoader(test_dataset, batch_size, num_workers=args.num_workers)

        tokenizer = test_dataset.get_tokenizer()

        model = Transformer(args.model, args.num_cls, text2text, num_layers=args.num_layers)
        model.eval()
        model.to(device)

        # Load model weights
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        data_len = test_dataset.__len__()
        print('Total Samples: {}'.format(data_len))

        # Inference
        metrics = compute_eval_metrics(model, loader, device, data_len, tokenizer, args, text2text, is_test=True)

        df = pd.DataFrame(metrics['meta'])
        df.to_csv(args.pred_file)

        # append predication to raw data
        append_to_data(args.test_file, args.pred_file, args.append_test_file)

        print(f'Results for model {args.model}')
        print(f'Results evaluated on file {args.test_file}')
        print('Sentence Accuracy: {:.6f}'.format(metrics['accuracy']))
    """

if __name__ == '__main__':
    main()
