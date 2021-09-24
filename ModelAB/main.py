import torch.nn as nn
import pandas as pd
from time import time
from model import ModelA, ModelB
from data import Com2SenseDataset
from config import config

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # Parse Args
    parser = config()
    args = parser.parse_args()

    # Multi-GPU
    device_ids = csv2list(args.gpu_ids, int)
    print('Selected GPUs: {}'.format(device_ids))

    # Device for loading dataset (batches)
    device = torch.device(device_ids[0]) if torch.cuda.is_available() else torch.device('cpu')
    if args.cpu:
        device = torch.device('cpu')

    # Train params
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    accumulation_steps = args.acc_step
    setup_seed(args.seed)
    # Todo: Verify the grad-accum code (loss avging seems slightly incorrect)

    # Train
    if args.mode == 'train':
        # Ensure CUDA available for training
        # assert torch.cuda.is_available(), 'No CUDA device for training!'

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
        train_datasets = Com2SenseDataset('train', tokenizer=args.model, data_dir=args.data_dir,
                                          max_seq_len=args.seq_len, template=args.template, mask_len=args.mask_len)

        valid_datasets = Com2SenseDataset('dev', tokenizer=args.model, data_dir=args.data_dir,
                                          max_seq_len=args.seq_len, template=args.template, mask_len=args.mask_len)

        train_loader = DataLoader(train_datasets, args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_datasets, args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=args.num_workers)

        # Tokenizer
        tokenizer = train_datasets.tokenizer

        # Split sizes
        train_size = train_datasets.__len__()
        valid_size = valid_datasets.__len__()
        log_msg = 'Train: {} \nValidation: {}\n\n'.format(train_size, valid_size)

        # Min of the total & subset size
        val_used_size = min(valid_size, args.val_size)
        log_msg += 'Validation Accuracy is computed using {} samples. See --val_size\n'.format(val_used_size)

        log_msg += 'No. of Classes: {}\n'.format(args.num_cls)
        print_log(log_msg, log_file)

        # Build Model
        if args.model_name == 'model_a':
            model = ModelA(prompt_model_name=args.model,
                           task_model_name=args.model,
                           num_cls=args.num_cls,
                           num_prompt_model_layer=-1,
                           num_task_model_layer=-1)
        elif args.model_name == 'model_b':
            model = ModelB(prompt_model_name=args.model,
                           task_model_name=args.model,
                           num_cls=args.num_cls,
                           num_prompt_model_layer=-1,
                           num_task_model_layer=-1,
                           is_projection=args.is_projection)
        else:
            raise NameError('Require model_a or model_b, but get neither.')

        model.to(device)
        model.train()

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        optimizer.zero_grad()

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

                with autocast(args.use_amp):
                    # Forward Pass
                    logits = model(batch)
                    label = batch['label']

                    # Compute Loss
                    loss = criterion(logits, label)

                # Backward Pass
                loss /= accumulation_steps
                scaler.scale(loss).backward()

                if curr_step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Print Results - Loss value & Validation Accuracy
                if curr_step % args.log_interval == 0:
                    # Validation set accuracy
                    if valid_datasets:
                        val_metrics = compute_eval_metrics(model, valid_loader, device, val_used_size, tokenizer)

                        # Reset the mode to training
                        model.train()

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
            if valid_datasets:
                log_msg = '-------------------------------------------------------------------------\n'
                val_metrics = compute_eval_metrics(model, valid_loader, device, valid_size, tokenizer)

                log_msg += '\nAfter {} epoch:\n'.format(epoch)
                log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                    val_metrics['accuracy'], val_metrics['loss'])

                # Save best model after every epoch
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]

                    step = '{:.1f}k'.format(curr_step / 1000) if curr_step > 1000 else '{}'.format(curr_step)
                    filename = 'ep_{}_stp_{}_acc_{:.4f}_{}.pth'.format(
                        epoch, step, best_val_acc, args.model.replace('-', '_').replace('/', '_'))

                    path = os.path.join(log_dir, filename)

                    model_state_dict = model.state_dict()
                    state_dict = {'model_state_dict': model_state_dict,
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg += "\n** Best Performing Model: {:.2f} ** \nSaving weights at {}\n".format(best_val_acc,
                                                                                                       path)

                log_msg += '-------------------------------------------------------------------------\n\n'
                print_log(log_msg, log_file)

                # Reset the mode to training
                model.train()

        writer.close()
        log_file.close()

    elif args.mode == 'test':

        # Dataloader
        test_datasets = Com2SenseDataset(args.test_file, tokenizer=args.model, data_dir=args.data_dir,
                                         max_seq_len=args.seq_len, template=args.template, mask_len=args.mask_len)
        test_loader = DataLoader(test_datasets, batch_size, shuffle=False, num_workers=args.num_workers)

        tokenizer = test_datasets.tokenizer

        if args.model_name == 'model_a':
            model = ModelA(prompt_model_name=args.model,
                           task_model_name=args.model,
                           num_cls=args.num_cls,
                           num_prompt_model_layer=-1,
                           num_task_model_layer=-1)
        elif args.model_name == 'model_b':
            model = ModelB(prompt_model_name=args.model,
                           task_model_name=args.model,
                           num_cls=args.num_cls,
                           num_prompt_model_layer=-1,
                           num_task_model_layer=-1,
                           example_file=args.run_name,
                           is_projection=args.is_projection)
        else:
            raise NameError('Require model_a or model_b, but get neither.')

        model.eval()
        model.to(device)

        # Load model weights
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        data_len = test_datasets.__len__()
        print('Total Samples: {}'.format(data_len))

        # Inference
        metrics = compute_eval_metrics(model, test_loader, device, data_len, tokenizer,
                                       is_pairwise=True, is_test=True)

        df = pd.DataFrame(metrics['meta'])
        df.to_csv(args.pred_file)

        print(f'Results for model {args.model}')
        print(f'Results evaluated on file {args.test_file}')
        print('Sentence Accuracy: {:.4f}'.format(metrics['accuracy']))
        print('Pairwise Accuracy: {:.4f}'.format(metrics['pair_acc']))


if __name__ == '__main__':
    main()
