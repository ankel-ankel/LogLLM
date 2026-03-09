from pathlib import Path
import os
import numpy as np
import torch

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
from tqdm import tqdm
from torch import nn
from model import LogLLM
from torch.utils.data import DataLoader
from customDataset import CustomDataset, CustomCollator, BalancedSampler
from torch import optim


n_epochs_1 = 1
n_epochs_2_1 = 1
n_epochs_2_2 = 1
n_epochs_3 = 2
dataset_name = 'BGL'  # 'Thunderbird' 'HDFS_v1' 'BGL'   'Liberty'
ft_dataset_name = 'HDFS' if dataset_name == 'HDFS_v1' else dataset_name
batch_size = 16
micro_batch_size = 1
gradient_accumulation_steps = batch_size // micro_batch_size

resume = False  # True = continue from last checkpoint | False = train from scratch

lr_1 = 5e-4
lr_2_1 = 5e-4
lr_2_2 = 5e-5
lr_3 = 5e-5
max_content_len = 100
max_seq_len = 64

ROOT_DIR = Path(__file__).resolve().parent
data_path = ROOT_DIR / 'data' / dataset_name / 'train.csv'

min_less_portion = 0.3

Bert_path = ROOT_DIR / 'models' / 'deberta-v3-large'
Llama_path = ROOT_DIR / 'models' / 'Meta-Llama-3.1-8B'

ft_path = ROOT_DIR / f'ft_model_{ft_dataset_name}'
ckpt_path = ft_path / 'checkpoint'
phase_file = ckpt_path / 'phase.txt'

# Phase order used for resume logic
PHASES = ['phase1', 'phase2_1', 'phase2_2', 'phase3']

device = torch.device("cuda:0")

print(f'n_epochs_1: {n_epochs_1}\n'
f'n_epochs_2_1: {n_epochs_2_1}\n'
f'n_epochs_2_2: {n_epochs_2_2}\n'
f'n_epochs_3: {n_epochs_3}\n'
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'micro_batch_size: {micro_batch_size}\n'
f'resume: {resume}\n'
f'lr_1: {lr_1}\n'
f'lr_2_1: {lr_2_1}\n'
f'lr_2_2: {lr_2_2}\n'
f'lr_3: {lr_3}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'min_less_portion: {min_less_portion}\n'
f'ft_path: {ft_path}\n'
f'device: {device}')

def print_number_of_trainable_model_parameters(model):
    params = set()
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            params.add(param)
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return params


def trainModel(model, dataloader, gradient_accumulation_steps, n_epochs, lr):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    trainable_model_params = print_number_of_trainable_model_parameters(model)
    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    normal_tokens = model.Llama_tokenizer('The sequence is normal.')['input_ids']
    anomalous_tokens = model.Llama_tokenizer('The sequence is anomalous.')['input_ids']
    special_normal_tokens = set(normal_tokens) - set(anomalous_tokens)
    special_anomalous_tokens = set(anomalous_tokens) - set(normal_tokens)

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)

    print(f'scheduler_step: {scheduler_step}')

    steps = 0
    for epoch in range(int(n_epochs)):
        total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        pbar = tqdm(dataloader, desc='Epoch {}/{}'.format(epoch, n_epochs))
        for i_th, batch_i in enumerate(pbar):
            steps += 1

            inputs = batch_i['inputs']
            seq_positions = batch_i['seq_positions']
            labels = batch_i['labels']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs, targets = model.train_helper(inputs, seq_positions, labels)

            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps

            loss.backward()

            if ((i_th + 1) % gradient_accumulation_steps == 0) or ((i_th + 1) == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            acc_mask = torch.zeros_like(targets,device=device).bool()
            for token in special_normal_tokens.union(special_anomalous_tokens):
                acc_mask[targets == token] = True

            total_acc += (outputs.argmax(1)[acc_mask] == targets[acc_mask]).sum().item()
            total_acc_count += acc_mask.sum()

            train_loss += loss.item() * gradient_accumulation_steps * targets.size(0)

            total_count += targets.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()
            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss = loss.item() * gradient_accumulation_steps)

            if steps % 10000 ==0:
                train_loss_epoch = train_loss / total_count
                train_acc_epoch = total_acc / total_acc_count
                print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                      f"[loss: {train_loss_epoch:3f}]"
                      f"[acc: {train_acc_epoch:3f}]")

                total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        if total_count > 0:
            train_loss_epoch = train_loss / total_count
            train_acc_epoch = total_acc / total_acc_count
            print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]"
                  f"[acc: {train_acc_epoch:3f}]")


def save_checkpoint(phase_name):
    model.save_ft_model(str(ckpt_path))
    phase_file.write_text(phase_name)
    print(f'Checkpoint saved after {phase_name}.')


def completed(phase_name):
    """Return True if this phase was already done in a previous run."""
    if not resume or not phase_file.exists():
        return False
    last = phase_file.read_text().strip()
    return PHASES.index(phase_name) <= PHASES.index(last)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project (4-bit bitsandbytes quantization).")
    for required_path in (data_path, Bert_path, Llama_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required path: {required_path}")

    # Decide whether to load from checkpoint or start fresh
    if resume and phase_file.exists():
        last_phase = phase_file.read_text().strip()
        print(f'*** Resuming from checkpoint (last completed: {last_phase}) ***')
        model = LogLLM(str(Bert_path), str(Llama_path), ft_path=str(ckpt_path), device=device,
                       max_content_len=max_content_len, max_seq_len=max_seq_len)
    else:
        if resume:
            print('*** resume=True but no checkpoint found — starting from scratch ***')
        else:
            print('*** Starting from scratch ***')
        model = LogLLM(str(Bert_path), str(Llama_path), device=device,
                       max_content_len=max_content_len, max_seq_len=max_seq_len)

    print(f'dataset: {data_path}')
    dataset = CustomDataset(str(data_path), drop_duplicates=False)

    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)

    # phase 1
    if not completed('phase1'):
        dataloader_max_samples = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            num_workers=4,
            sampler=BalancedSampler(dataset, target_ratio=min_less_portion, max_samples=1000),
            collate_fn=collator,
            drop_last=True
        )
        print("*" * 10 + "Start training Llama" + "*" * 10)
        model.set_train_only_Llama()
        trainModel(model, dataloader_max_samples, gradient_accumulation_steps, n_epochs_1, lr_1)
        del dataloader_max_samples
        torch.cuda.empty_cache()
        save_checkpoint('phase1')
    else:
        print('Skipping phase1 (already done)')

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=4,
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion),
        collate_fn=collator,
        drop_last=True
    )

    # phase 2-1
    if not completed('phase2_1'):
        print("*" * 10 + "Start training projector" + "*" * 10)
        model.set_train_only_projector()
        trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_1, lr_2_1)
        torch.cuda.empty_cache()
        save_checkpoint('phase2_1')
    else:
        print('Skipping phase2_1 (already done)')

    # phase 2-2
    if not completed('phase2_2'):
        print("*" * 10 + "Start training projector and Bert" + "*" * 10)
        model.set_train_projectorAndBert()
        trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_2, lr_2_2)
        torch.cuda.empty_cache()
        save_checkpoint('phase2_2')
    else:
        print('Skipping phase2_2 (already done)')

    # phase 3
    if not completed('phase3'):
        model.set_finetuning_all()
        print("*" * 10 + "Start training entire model" + "*" * 10)
        trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_3, lr_3)
        save_checkpoint('phase3')
    else:
        print('Skipping phase3 (already done)')

    # Final save to ft_path (the "production" checkpoint)
    model.save_ft_model(str(ft_path))
    print(f'Training complete. Model saved to {ft_path}')
