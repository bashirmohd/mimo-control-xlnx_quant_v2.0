import argparse
import datetime
import json
import numpy as np
from scipy.stats import special_ortho_group
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from recognizer import create_recognizer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gym'))
from laser_cbc.physics.model import CBCModel


class CombiningDataGen(Dataset):
    '''
    len:            number of samples in dataset
    dither_range_deg:      training phase range in degrees / 2
    double_frame:   use difference phase mapping to double pattern
    '''
    def __init__(self, **kwargs):
        self.dither_range_deg = kwargs.get('dither_range_deg', 45)
        self.double_frame = kwargs.get('double_frame', False)
        self.ortho_sampling = kwargs.get('ortho_sampling', False)
        self.len = kwargs.get('n_samples', 1000)
        self.model = CBCModel(**kwargs)
        self.rng = np.random.default_rng()
        x, y = self.gen_data_set()
        self.X = torch.from_numpy(x)
        self.Y = torch.from_numpy(y)

    def __getitem__(self, idx):
        'Generates one sample of data'
        return self.X[idx], self.Y[idx]

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def gen_random_sample(self, i):
        '''
        Generate random samples in N dimensional phase space by phase
        perturbing the system at range [-1, 1] * self.dither_range_deg.

        Use random, or special orthogonal group for more efficient mapping:
        orthognoally sampling N times, then random rotate.
        see: https://en.wikipedia.org/wiki/Orthogonal_group#SO(n)
        '''
        if not self.double_frame:
            self.model.reset()  # perturb system from optimal
        pattern0 = self.model.nonzero_pattern
        if self.ortho_sampling:
            idx = i % self.model.n_beams
            if idx == 0:
                self.ortho_samples = special_ortho_group.rvs(self.model.n_beams)
            phs_perturb_deg = self.ortho_samples[idx] * self.dither_range_deg
        else:
            phs_perturb_deg = (self.rng.random(self.model.n_beams) * 2 - 1) * self.dither_range_deg
        pattern1 = self.model.perturb_phase_arr(phs_perturb_deg)
        pattern = np.concatenate((pattern0, pattern1)) if self.double_frame else pattern1
        return pattern, phs_perturb_deg

    def normalize(self, array):
        mean, std = np.mean(array, axis=0), np.std(array, axis=0)
        return (array - mean) / std

    def gen_data_set(self):
        '''
        Generate training dataset as pair of x: pattern, y: phase error
        '''
        x_size = self.model.n_pattern_beams
        x_size = 2 * x_size if self.double_frame else x_size
        y_size = self.model.n_beams
        x_set = np.empty((self.len, x_size), dtype=np.float32)
        y_set = np.empty((self.len, y_size), dtype=np.float32)
        for i in range(self.len):
            x_set[i], y_set[i] = self.gen_random_sample(i)
        self.config = self.get_normalize_config(x_set, y_set)
        x = self.normalize(x_set)
        y = self.normalize(y_set)
        return x, y

    def get_normalize_config(self, x, y):
        config = {}
        config['mu_X'] = np.mean(x, axis=0).ravel().tolist()
        config['mu_Y'] = np.mean(y, axis=0).ravel().tolist()
        config['sigma_X'] = np.std(x, axis=0).ravel().tolist()
        config['sigma_Y'] = np.std(y, axis=0).ravel().tolist()
        return config


class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience.'''
    def __init__(self, patience=7, verbose=False, delta=0):
        '''
        patience (int): How long to wait after last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def export_np_array(arr_in, double_frame, net_config):
    print("Exporting one dataset sample for FPGA inference testing...")
    arr_tmp =  np.array(arr_in, dtype=np.float32)
    if double_frame is True:
        arr_tmp.shape = (1, 1, 1, 1, 50)
    else:
        arr_tmp.shape = (1, 1, 1, 1, 25)
    np.save('nn_trained/sample.npy', arr_tmp, allow_pickle=True, fix_imports=True)


def train_model(model, dataset, device='cpu',
                patience=7, n_epochs=50, batch_size=64, verbose=False, export_frame=False):
    model.to(device)
    df = pd.DataFrame(columns=['train_loss', 'valid_loss'])
    sigma_y = np.average(dataset.config['sigma_Y'])

    vali_num = int(0.1 * len(dataset))
    train_num = len(dataset) - vali_num
    train_dataset, vali_dataset = random_split(dataset, [train_num, vali_num])

    if export_frame is True:
        export_np_array(vali_dataset[0][0], config['double_frame'], config['net_config'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training

        for batch, (x, y) in enumerate(train_loader, 0):
            # get the inputs; data is a tuple of (input, target)
            inputs = x.float().to(device)
            targets = y.float().to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            # calculate the loss
            loss = torch.sqrt(criterion(outputs, targets)) * sigma_y
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for (x, y) in valid_loader:
            inputs = x.float().to(device)
            targets = y.float().to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            # calculate the loss
            loss = torch.sqrt(criterion(outputs, targets)) * sigma_y
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        df.loc[epoch] = [train_loss, valid_loss]

        epoch_len = len(str(n_epochs))

        print(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
              + f'train_loss: {train_loss:4.2f} deg, valid_loss: {valid_loss:4.2f} deg')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print('Early stopping')
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, df


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-n', '--n_samples', type=int, default=5000, help='n_samples per beam')
    p.add_argument('-m', type=int, default=9, choices=[3, 9], help='MxM beam shape')
    p.add_argument('-e', '--n_epochs', type=int, default=50, help='number of episodes')
    p.add_argument('--weight', help='output NN weight file')
    p.add_argument('--force_cpu', action='store_true', help='Forces CPU usage')
    p.add_argument('--test_3_in_9', action='store_true')
    p.add_argument('--double_frame', action='store_true', help='doubled frame training')
    p.add_argument('--ortho_sampling', action='store_true', help='use random orthogonal sampling')
    p.add_argument("--rms_measure_noise", type=float, default=0.1, help="rms camera noise")
    p.add_argument("--phs_drift_step_deg", type=float, default=5, help="phs_drift_step_deg")
    p.add_argument("--dither_range_deg", type=int, default=45, help="dither phase range")
    p.add_argument("--net_config", type=str, default="3x3", help="config in recognizer.py")
    args = p.parse_args()
    device = 'cpu' if args.force_cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = {
        'M': args.m,
        'verbose': args.verbose,
        'n_samples': args.n_samples,
        'test_3_in_9': args.test_3_in_9,
        'double_frame': args.double_frame,
        'ortho_sampling': args.ortho_sampling,
        'rms_measure_noise': args.rms_measure_noise,
        'phs_drift_step_deg': args.phs_drift_step_deg,
        'dither_range_deg': args.dither_range_deg,
        'net_config' : args.net_config
    }

    json_file = json.dumps(config)
    fout = 'nn_trained/{}by{}_{}deg_config.json'.format(args.m, args.m, args.dither_range_deg)
    with open(fout, "w") as jsonfile:
        jsonfile.write(json_file)
        print("Write successful")

    if args.weight is None:
        fout = 'nn_trained/{}by{}_{}deg.pth'.format(args.m, args.m, args.dither_range_deg)
    else:
        fout = args.weight
    print('Training model ' + fout + '...')

    net = create_recognizer(args.double_frame, args.net_config)

    print(net)
    dataset = CombiningDataGen(**config)
    with open(fout.replace('pth', 'json'), 'w+') as f:
        json.dump(dataset.config, f, indent=4)
    print('Start Training...')
    start = datetime.datetime.now()

    net_trained, df = train_model(
        net, dataset, device,
        n_epochs=args.n_epochs, verbose=args.verbose, export_frame=True)
    # torch.save(net_trained.state_dict(), fout)
    torch.save(net_trained, fout)
    df.to_csv(fout.replace('pth', 'csv'), index=False)
    # save dataset
    torch.save(dataset, fout.replace('.pth', '_dat.pt'))

    end = datetime.datetime.now()
    print('Finished Training, wrote {}'.format(fout))
    elapsed_sec = round((end - start).total_seconds(), 2)
    print('It took', elapsed_sec, 'seconds to Train.')
