import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import pytorch_lightning as pl
import torch.cuda

from models.densenet import DenseNet
from models.resnet import ResNet
from dataset.cifar10 import CIFAR10

from injector import Injector
from extractor import Extractor
from extractor_callback import ExtractorCallback

from logger.csv_logger import CSVLogger

from analyze_distributions import get_digit_distribution, save_data

import logging

import warnings

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# Filter TiffImagePlugin warnings
warnings.filterwarnings("ignore")

# remove PIL debugging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.CRITICAL)

# A logger for generic events
log = logging.getLogger()
log.setLevel(logging.DEBUG)

logging.basicConfig(filename='maleficnet.log', level=logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def weights_init_normal(m):
    classname = m.__class__.__name__
    state_dict = m.state_dict()

    if classname.find('Linear') != -1:
        if 'weight' in state_dict.keys():
            weights = state_dict['weight'].detach().cpu().numpy().flatten()
            mean = np.mean(weights)
            std = np.std(weights)
        else:
            y = m.in_features
            mean = 0.0
            std = 1 / np.sqrt(y)

        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0)


def initialize_model(model_name, dim, num_classes, only_pretrained):
    model = None

    if model_name == "densenet":
        model = DenseNet(input_shape=dim,
                         num_classes=num_classes,
                         only_pretrained=only_pretrained,
                         model_size=169)
    elif model_name == "resnet":
        model = ResNet(input_shape=dim,
                       num_classes=num_classes,
                       only_pretrained=only_pretrained)

    return model


def main(gamma, model_name, dataset, epochs, dim, num_classes, batch_size, num_workers, payload, only_pretrained, fine_tuning, chunk_factor, seed):
    # checkpoint path
    checkpoint_path = Path(os.getcwd()) / 'checkpoints'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    pre_model_name = checkpoint_path / f'{model_name}_{dataset}_pre_model.pt'
    post_model_name = checkpoint_path / \
        f'{model_name}_{dataset}_clean_{"T" if only_pretrained else "F"}_{seed}_model.pt'

    message_length, malware_length, hash_length = None, None, None

    # Init logger
    logger = CSVLogger('train.csv', 'val.csv', ['epoch', 'loss', 'accuracy'], [
        'epoch', 'loss', 'accuracy'])

    # Init our data pipeline
    if dataset == 'cifar10':
        data = CIFAR10(base_path=Path(os.getcwd()),
                       batch_size=batch_size,
                       num_workers=num_workers)
    
    model = initialize_model(model_name, dim, num_classes, only_pretrained)
    model.apply(weights_init_normal)

    
    

    if not fine_tuning:
        trainer = pl.Trainer(max_epochs=epochs + 15)
        
        if not only_pretrained:
            # Train the model only if we want to save a new one! 🚆
            trainer.fit(model, data)
        else:
            model.load_state_dict(torch.load(pre_model_name))

        # Test the model
        trainer.test(model, data)


        del trainer

        # Create a new trainer
        trainer = pl.Trainer(max_epochs=epochs)

        # Train a few more epochs to restore performances 🚆
        trainer.fit(model, data)

        # Test the model again
        results = trainer.test(model, data)
        with open("digit_analysis/acc_log.txt", "a") as test_acc_file:
            test_acc_file.write(f"{post_model_name}: {results}\n")

        torch.save(model.state_dict(), post_model_name)
        
        del trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maleficnet Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='The dataset to use: cifar10')
    parser.add_argument('--dim', type=int, default=32,
                        help='The dataset dimension to use: 32 (CIFAR10) or 224 (IMAGENET)')
    parser.add_argument('--model', '-m', default='vgg11', type=str,
                        help='Name of the model: [densenet]')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='Number of classes (e.g., 10 if dataset is CIFAR10).')
    parser.add_argument('--only_pretrained', default=False, action='store_true',
                        help='Whether to use a only pretrained model or not.')
    parser.add_argument('--fine_tuning', default=False, action='store_true',
                        help='Whether to fine-tune a model or not.')
    parser.add_argument('--epochs', type=int, default=60,
                        help='The number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Input batch size')
    parser.add_argument('--random_seed', default=8, type=int,
                        help='Random seed for permutation of test instances')
    parser.add_argument('--num_workers', default=3, type=int,
                        help='The number of concurrent processes to parse the dataset.')
    parser.add_argument('--payload', type=str, default='payload.exe',
                        help='The payload to inject in the model.')
    parser.add_argument('--gamma', type=float, default=0.0009,
                        help='The gamma used to inject.')

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)

    main(gamma=args.gamma,
         model_name=args.model,
         dataset=args.dataset,
         epochs=args.epochs,
         dim=args.dim,
         num_classes=args.num_classes,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         payload=args.payload,
         only_pretrained=args.only_pretrained,
         fine_tuning=args.fine_tuning,
         chunk_factor=6,
         seed = args.random_seed)
