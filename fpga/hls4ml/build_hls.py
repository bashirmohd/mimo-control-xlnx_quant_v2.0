import argparse
import os
import hls4ml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# CombiningDataGen is needed for unpacking saved dataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'feedback'))
from train_nn import CombiningDataGen

# https://forums.xilinx.com/t5/SDSoC-Environment-and-reVISION/SDx-2016-4-HLS-compilation-errors-from-stdio-h/td-p/766447
# sudo apt-get install gcc-multilib g++-multilib


def compile_hls_model(nn_file, output_dir):
    # load model
    config = hls4ml.utils.example_models._create_default_config(nn_file, 'PytorchModel')
    config['OutputDir'] = output_dir

    # convert
    hls_model = hls4ml.converters.convert_from_config(config)

    # compile (optional)
    hls_model.compile()
    return hls_model


def test_hls_model(model_float, hls_model, dataset, device='cpu'):
    test_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    quant_losses = []
    float_losses = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for (x, y) in test_loader:
            inputs = x.float().to(device)
            targets = y.float().to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model_float(inputs)
            # calculate the loss
            loss = torch.sqrt(criterion(outputs, targets))
            float_losses.append(loss.item())

            y_hls = hls_model.predict(x.numpy())
            loss_quant = torch.sqrt(criterion(torch.from_numpy(y_hls), targets))
            quant_losses.append(loss_quant.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        quant_loss = np.average(quant_losses)
        float_loss = np.average(float_losses)

        print(f'float model loss: {float_loss:4.4f}')
        print(f'quant model loss: {quant_loss:4.4f}')


if __name__ == '__main__':
    # https://www.xilinx.com/support/answers/69355.html
    os.environ['LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'

    p = argparse.ArgumentParser()
    p.add_argument(
        '--nn_file',
        help='pre-trained nn weight file *.pth',
        default='../../feedback/nn_trained/3by3_30deg.pth')
    args = p.parse_args()
    nn_file = args.nn_file
    output_dir = os.path.join('build', os.path.splitext(os.path.basename(nn_file))[0])

    print('   Compiling model...')
    hls_model = compile_hls_model(nn_file, output_dir)

    print('   Check performance...')
    dataset = torch.load(nn_file.replace('.pth', '_dat.pt'))
    model_float = torch.load(nn_file, map_location=torch.device('cpu'))
    test_hls_model(model_float, hls_model, dataset)

    print('   Build and synthesize...')
    # build using Vivado HLS
    hls_model.build()

    # Print out the report if you want
    hls4ml.report.read_vivado_report(output_dir)
