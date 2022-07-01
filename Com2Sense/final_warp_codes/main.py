import os
import copy
import sys
import torch
import torch.nn as nn
import csv
import argparse
import pandas as pd
from time import time
from model import Transformer
from transformers import AutoTokenizer
from dataloader import BaseDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import ExponentialLR

#重新设置，使得模型从头到尾都为eval状态

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
    parser.add_argument('--mode',           type=str,       help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir',       type=str,       help='root directory to save model & summaries')
    parser.add_argument('--expt_name',      type=str,       help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name',       type=str,       help='expt_dir/expt_name/run_name: organize training runs')
    parser.add_argument('--test_file', type=str, default = 'test', help = 'The file containing test data to evaluate in test mode.')

    # Model params
    parser.add_argument('--model',          type=str,       help='transformer model (e.g. roberta-base)', required=True)
    parser.add_argument('--num_layers', type = int, help = 'Number of hidden layers in transformers (default number if not provided)', default=-1)
    parser.add_argument('--seq_len',        type=int,       help='tokenized input sequence length', default=256)
    parser.add_argument('--num_cls',        type=int,       help='model number of classes', default=2)
    parser.add_argument('--ckpt',           type=str,       help='path to model checkpoint .pth file')

    # Data params
    parser.add_argument('--pred_file',      type=str,       help='address of prediction csv file, for "test" mode', default='results.csv')
    parser.add_argument('--dataset',        type=str,       help='list of datasets seperated by commas', required=True)

    # Training params
    parser.add_argument('--lr',             type=float,     help='learning rate', default=1e-5)
    parser.add_argument('--epochs',         type=int,       help='number of epochs', default=100)
    parser.add_argument('--batch_size',     type=int,       help='batch size', default=8)
    parser.add_argument('--acc_step',       type=int,       help='gradient accumulation steps', default=1)
    parser.add_argument('--log_interval',   type=int,       help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval',  type=int,       help='save model after `n` weight update steps', default=30000)
    parser.add_argument('--val_size',       type=int,       help='validation set size for evaluating metrics', default=2048)
    parser.add_argument('--seed', type=int, help='validation set size for evaluating metrics', default=808)

    # GPU params
    parser.add_argument('--gpu_ids',        type=str,       help='GPU IDs (0,1,2,..) seperated by comma', default='3')
    parser.add_argument('-data_parallel',       help='Whether to use nn.dataparallel (currently available for BERT-based models)', action = 'store_true')
    parser.add_argument('--use_amp',        type=str2bool,  help='Automatic-Mixed Precision (T/F)', default='T')
    parser.add_argument('-cpu',       help='use cpu only (for test)', action = 'store_true')

    # Misc params
    parser.add_argument('--num_workers',    type=int,       help='number of worker threads for Dataloader', default=1)

    # Parse Args
    args = parser.parse_args()

    # Random seed
    setup_seed(args.seed) 

    # Dataset list
    dataset_names = csv2list(args.dataset)

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
    # Todo: Verify the grad-accum code (loss avging seems slightly incorrect)

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
        dataset = BaseDataset('train', tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text, uniqa = uniqa)
        train_datasets = dataset.concat(dataset_names)

        dataset = BaseDataset('dev', tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text, uniqa = uniqa)
        val_datasets = dataset.concat(dataset_names)

        train_loader = DataLoader(train_datasets, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_datasets, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        # In multi-dataset setups, also track dataset-specific loaders for validation metrics
        val_dataloaders = []
        if len(dataset_names) > 1:
            for val_dset in val_datasets.datasets:
                loader = DataLoader(val_dset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

                val_dataloaders.append(loader)

        # Tokenizer
        tokenizer = dataset.get_tokenizer()

        # Split sizes
        train_size = train_datasets.__len__()
        val_size = val_datasets.__len__()
        log_msg = 'Train: {} \nValidation: {}\n\n'.format(train_size, val_size)

        # Min of the total & subset size
        val_used_size = min(val_size, args.val_size)
        log_msg += 'Validation Accuracy is computed using {} samples. See --val_size\n'.format(val_used_size)

        log_msg += 'No. of Classes: {}\n'.format(args.num_cls)
        print_log(log_msg, log_file)

        # Build Model
        model = Transformer(args.model, args.num_cls, text2text, device_ids,num_layers = args.num_layers)
        if args.data_parallel and  not args.ckpt:
             model = nn.DataParallel(model, device_ids = device_ids)
             device = torch.device(f'cuda:{model.device_ids[0]}')
       
        if not model.parallelized:
             model.to(device)
 
        if type(model) != nn.DataParallel:
            if not model.parallelized:
               model.to(device)
       ## model.train()
        model.eval()


        prompts=[]      #included 2 kinds of prompt tokens: 15 input prompts + 2 output prompts + 2 bias for output prompts + 1 fixed mask token
        temp=model.hand_embed(torch.tensor([50264]).to(device))[0].tolist()     #Initialize with <mask> token  (input prompt)
        for num_t in range(16):
###            temp=torch.empty([1,1024]).to(device).requires_grad_()
           # tem=torch.tensor(temp).to(device).requires_grad_()   #In this way , is_leaf==TRUE
            tem=torch.randn([1024]).to(device).requires_grad_()
            ###torch.nn.init.kaiming_normal_(temp)
  #          print(temp[0][:5])
            ###temp=temp[0].tolist()
            ###tem=torch.tensor(temp).to(device).requires_grad_()
 #           print(tem)
#            print(tem.is_leaf)
            prompts.append(tem)

        temp=model.hand_embed(torch.tensor([1593]).to(device))[0].tolist()    #wrong token
        tem=torch.tensor(temp).to(device).requires_grad_()
        prompts.append(tem)

        tem=torch.tensor(0.0).to(device).requires_grad_()    #bias_1
        prompts.append(tem)

        temp=model.hand_embed(torch.tensor([4070]).to(device))[0].tolist()    #right token
        tem=torch.tensor(temp).to(device).requires_grad_()
        prompts.append(tem)

        tem=torch.tensor(0.0).to(device).requires_grad_()         #bias_2
        prompts.append(tem)



        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(prompts, lr,weight_decay=1)    #加了正则化,正则化和scheduler是同时加上的
        scheduler=ExponentialLR(optimizer,gamma=0.95)    #加了指数scheduler
        optimizer.zero_grad()
        model.zero_grad()

        scaler = GradScaler(enabled=args.use_amp)

        # Step & Epoch
        start_epoch = 1
        curr_step = 1
        best_val_acc = 0.0

        # Load model checkpoint file (if specified)
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)

            # Load model & optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.data_parallel:
                model = nn.DataParallel(model, device_ids = device_ids)
                device = torch.device(f'cuda:{model.device_ids[0]}')
            model.to(device)

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
  

                embeds=model.hand_embed(batch['tokens'])
                embeddings=torch.zeros(batch_size,args.seq_len+16,1024).to(device)
                atten=torch.zeros(batch_size,args.seq_len+16,dtype=int).to(device)
                zero=torch.zeros(1024).to(device)

                for num_emb in range(batch_size):
                    temp=embeds[num_emb]    #[L,D]
                    temp_att=batch['attn_mask'][num_emb]    #[L]
     #               embedding=embeddings[num_emb]   #[L+16 , D]
      #              att=atten[num_emb]     #[L+16]
                    for ii in range(args.seq_len+16):
                        if ii==0:
                            embeddings[num_emb][ii]=temp[ii]
                            atten[num_emb][ii]=temp_att[ii]
                        elif 1<=ii<=5:
                            embeddings[num_emb][ii]=zero+prompts[ii-1]
                            atten[num_emb][ii]=torch.tensor(1).to(device)
                        elif 6<=ii<=19:
                            embeddings[num_emb][ii]=temp[ii-5]
                            atten[num_emb][ii]=temp_att[ii-5]
                        elif 20<=ii<=25:
                            embeddings[num_emb][ii]=zero+prompts[ii-15]
                            atten[num_emb][ii]=torch.tensor(1).to(device)
                        elif 26<=ii<=args.seq_len+10:
                            embeddings[num_emb][ii]=temp[ii-11]
                            atten[num_emb][ii]=temp_att[ii-11]
                        else:
                            embeddings[num_emb][ii]=zero+prompts[ii-args.seq_len]
                            atten[num_emb][ii]=torch.tensor(1).to(device)

                with autocast(args.use_amp):
                    if text2text:
                        # Forward + Loss
                        output = model(batch)
                        loss = output[0]

                    else:
                        # Forward Pass
                        outs = model(atten,embeddings)
                        label_logits=torch.zeros(batch_size,2).to(device)
                        
                        label_logits[:,0]=(outs*prompts[16]).sum(1)+prompts[17]
                        label_logits[:,1]=(outs*prompts[18]).sum(1)+prompts[19]
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
                    model.zero_grad()
                     

                # Print Results - Loss value & Validation Accuracy
                if curr_step % args.log_interval == 0:
                    # Validation set accuracy
                    temp_prompt=copy.deepcopy(prompts)
                    if val_datasets:
                        val_metrics = compute_eval_metrics(temp_prompt,batch_size,args.seq_len,model, val_loader, device, val_used_size, tokenizer,text2text, parallel = args.data_parallel)

                        # Reset the mode to training
                        ##model.train()
                        model.eval()

                        log_msg = 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}'.format(
                                val_metrics['accuracy'], val_metrics['loss'])

                        print_log(log_msg, log_file)

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Loss', val_metrics['loss'], curr_step)
                        writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h |'.format(
                            epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed)

                    print_log(log_msg, log_file)

                # Save the model
                if curr_step % args.save_interval == 0:
                    path = os.path.join(log_dir, 'model_' + str(curr_step) + '.pth')

                    state_dict = {'model_state_dict': model.state_dict(),
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg = 'Saving the model at the {} step to directory:{}'.format(curr_step, log_dir)
                    print_log(log_msg, log_file)

                curr_step += 1
     
            # Validation accuracy on the entire set
            scheduler.step()
            temp_prompt=copy.deepcopy(prompts)
            if val_datasets:
                log_msg = '-------------------------------------------------------------------------\n'
                val_metrics = compute_eval_metrics(temp_prompt,batch_size,args.seq_len,model, val_loader, device, val_size, tokenizer, text2text, parallel = args.data_parallel)

                log_msg += '\nAfter {} epoch:\n'.format(epoch)
                log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                    val_metrics['accuracy'], val_metrics['loss'])

                # For Multi-Dataset setup:
                if len(dataset_names) > 1:
                    # compute validation set metrics on each dataset independently
                    for loader in val_dataloaders:
                        metrics = compute_eval_metrics(model, loader, device, val_size, tokenizer, text2text, parallel = args.data_parallel)

                        log_msg += '\n --> {}\n'.format(loader.dataset.get_classname())
                        log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                            metrics['accuracy'], metrics['loss'])

                # Save best model after every epoch
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]

                    step = '{:.1f}k'.format(curr_step/1000) if curr_step > 1000 else '{}'.format(curr_step)
                    filename = 'ep_{}_stp_{}_acc_{:.4f}_{}.pth'.format(
                        epoch, step, best_val_acc, args.model.replace('-', '_').replace('/','_'))
                    p_name = 'prompts_ep_{}_stp_{}_acc_{:.4f}_{}.pth'.format(
                        epoch, step, best_val_acc, args.model.replace('-', '_').replace('/','_'))
                    

                    path = os.path.join(log_dir, filename)
                    path_p = os.path.join(log_dir,p_name)
                    if args.data_parallel:
                      model_state_dict = model.module.state_dict()
                    else:
                      model_state_dict = model.state_dict()
                    state_dict = {'model_state_dict': model_state_dict,
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)
                    p_save=copy.deepcopy(prompts)
                    #p_save=torch.tensor(np.array([item.cpu().detach().numpy() for item in prompt_save],dtype=np.float64)).to(device)
                    
                    torch.save(p_save,path_p)

                    log_msg += "\n** Best Performing Model: {:.2f} ** \nSaving weights at {}\n".format(best_val_acc, path)

                log_msg += '-------------------------------------------------------------------------\n\n'
                print_log(log_msg, log_file)

              
                ##model.train()
                model.eval()

        writer.close()
        log_file.close()

    elif args.mode == 'test':

        # Dataloader
        dataset = BaseDataset(args.test_file, tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text, uniqa = uniqa)
        datasets = dataset.concat(dataset_names)

        loader = DataLoader(datasets, batch_size, num_workers=args.num_workers)

        tokenizer = dataset.get_tokenizer()

        model = Transformer(args.model, args.num_cls, text2text, num_layers = args.num_layers)
        model.eval()
        model.to(device)

        # Load model weights
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        data_len = datasets.__len__()
        print('Total Samples: {}'.format(data_len))

        is_pairwise = 'com2sense' in dataset_names

        # Inference
        #得到prompts:
        
        pre_prompts=torch.load("/nas/home/ruosongy/warp/Com2Sense/results_log/com2sense/roberta_large/warp_32_eq_1e3/prompts_ep_99_stp_9.9k_acc_53.2500_roberta_large.pth",map_location=device)
        prompts=[]
        for nnn in range(20):
            
            prompts.append(pre_prompts[nnn].to(device))


        temp_prompt=copy.deepcopy(prompts)

        metrics = compute_eval_metrics(temp_prompt,batch_size,args.seq_len,model, loader, device, data_len, tokenizer,text2text, is_pairwise =is_pairwise, is_test=True, parallel = args.data_parallel)

        df = pd.DataFrame(metrics['meta'])
        df.to_csv(args.pred_file)

        print(f'Results for model {args.model}')
        print(f'Results evaluated on file {args.test_file}')
        print('Sentence Accuracy: {:.4f}'.format(metrics['accuracy']))
        if is_pairwise:
            print('Pairwise Accuracy: {:.4f}'.format(metrics['pair_acc']))


if __name__ == '__main__':
    main()
