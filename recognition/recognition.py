import os
import argparse
import yaml
import shutil
from typing import List, int
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from recognition.util import encode, decode
from recognition.model import CRNN, STRDataset
from recognition.process import vocab
from recognition.process.preprocessing import *
from recognition.train import *

def arguments():
    arg = argparse.ArgumentParser(add_help= False)
    arg.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu')
    
    arg.add_argument("--seed", default = 0, type = int)
    arg.add_argument("--val_size", default = 0.2)
    arg.add_argument("--test_size", default = 0.125)
    arg.add_argument('--is_shuffle', default = True)
    arg.add_argument('--train_batch_size', default = 32)
    arg.add_argument('--test_batch_size', default = 8)
    arg.add_argument('--hidden_size', default = 256)
    arg.add_argument('--n_layers', default =2)
    arg.add_argument('--dropout_prob', default = 0.3)
    arg.add_argument('--unfreeze_layers', default = 3)


    arg.add_argument('--epochs', default = 100)
    arg.add_argument('--lr', default = 0.001)
    arg.add_argument('--weight_decay', default = 1e-5)
    arg.add_argument('--scheduler_step_size', default = None, help = 'by default it should equals half of epoches')
    arg.add_argument('--gamma', default =0.1)
    return arg

def main(arg):
    arg = arguments()
    device=  arg.device
    if arg.scheduler_step_size is None:
        arg.scheduler_step_size = arg.epochs *0.5

    X_train, X_val, y_train, y_val = train_test_split(
                                                    vocab.img_paths, vocab.labels,
                                                    test_size= arg.val_size,
                                                    random_state = arg.seed,
                                                    shuffle= arg.is_shuffle
                                                )

    X_train, X_test, y_train, y_test = train_test_split(
                                                    X_train, y_train,
                                                    test_size= arg.test_size,
                                                    random_state= arg.seed,
                                                    shuffle = arg.is_shuffle
                                                    )
    train_dataset = STRDataset(
                                X_train, y_train,
                                char_to_idx= vocab.char_to_idx,
                                max_label_len= vocab.max_label_len,
                                label_encoder= encode,
                                transform=data_transforms['train']
                            )
    val_dataset = STRDataset(
                                X_val, y_val,
                                char_to_idx = vocab.char_to_idx,
                                max_label_len = vocab.max_label_len,
                                label_encoder=encode,
                                transform=data_transforms['val']
                            )
    test_dataset = STRDataset(
                                X_test, y_test,
                                char_to_idx = vocab.char_to_idx,
                                max_label_len = vocab.max_label_len,
                                label_encoder = encode,
                                transform=data_transforms['val']
                            )
    train_loader = DataLoader(
                                train_dataset,
                                batch_size = vocab.train_batch_size,
                                shuffle=True
                            )
    val_loader = DataLoader(
                                val_dataset,
                                batch_size= arg.test_batch_size,
                                shuffle=False
                            )
    test_loader = DataLoader(
                                test_dataset,
                                batch_size = arg.test_batch_size,
                                shuffle=False
                            )
    train_features, train_labels, train_lengths = next(iter(train_loader))
    model = CRNN(
                vocab_size= vocab.vocab_size,
                hidden_size= arg.hidden_size,
                n_layers= arg.n_layers,
                dropout= arg.dropout_prob,
                unfreeze_layers= arg.unfreeze_layers
                ).to(device)
    criterion = nn.CTCLoss(
                            blank=vocab.char_to_idx[vocab.blank_char], 
                            zero_infinity=True
                        )
    optimizer = torch.optim.Adam(
                                model.parameters(), 
                                lr= arg.lr, 
                                weight_decay= arg.weight_decay
                            )
    scheduler = torch.optim.lr_scheduler.StepLR(
                                                optimizer, 
                                                step_size= arg.scheduler_step_size, 
                                                gamma= arg.gamma
                                            )
    train_losses, val_losses = fit(
                                    model,
                                    train_loader,
                                    val_loader,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    device,
                                    arg.epochs
                                )
    return train_losses, val_losses, model

if __name__ == "__main__":
    arg  = arguments()
    main(arg)
    train_losses, val_losses, model = main(arg)
    save_model_path = 'models/ocr_crnn_resnet_best.pt'
    torch.save(
        model.state_dict(), 
        save_model_path
    )

