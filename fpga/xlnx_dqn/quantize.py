import sys
import os
import io
import json
import argparse
import torch
import pandas as pd
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer

from torch.utils.data import random_split

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'feedback'))
from recognizer import create_recognizer
from train_nn import CombiningDataGen, train_model

DIVIDER = '-----------------------------------------'


def quantize(dataset, model, weight, dim, double_frame, quant_mode, batchsize, quant_model, device):
    # weight-loading first
    # model.load_state_dict(torch.load(weight, map_location=device))
    model = torch.load(weight, map_location=device)
    print(model)

    # override batchsize if in test mode
    if quant_mode == 'test':
        batchsize = 1

    # if one has RuntimeError: size mismatch, m1: [a x b], m2: [c x d]
    # all they should care is b=c
    if dim == 3 and double_frame == True:
        b = 50
    elif dim == 3 and double_frame == False:
        b = 25
    else:
        # debatable
        b = 242

    rand_in = torch.randn([batchsize, b])

    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model)
    quantized_model = quantizer.quant_model

    # re-train?
    print("")
    print(DIVIDER)
    print("Model successfully quantized. Evaluating quantized model...")
    print(DIVIDER)
    print("")
    train_model(quantized_model, dataset, device, n_epochs=40, verbose=False)

    ##################
    ### DEPRECATED ###
    ##################
    # vali_num = int(0.1 * len(dataset))
    # print('vali_num = ', str(vali_num))
    # print('len(dataset) = ', str(len(dataset)))
    # train_num = len(dataset) - vali_num
    # print('train_num = ', str(train_num))
    # train_dataset, vali_dataset = random_split(dataset, [train_num, vali_num])

    # test_loader = torch.utils.data.DataLoader(vali_dataset,
    #                                           batch_size=batchsize,
    #                                           shuffle=True)

    # # evaluate
    # try:
    #     # test(quantized_model, device, test_loader)
    #     # evaluate_model(quantized_model, test_loader, device, nn.MSELoss())
    #     eval_quant_laser(quantized_model, device, test_loader, nn.MSELoss())
    # except ValueError:
    #     raise RuntimeError("Failed to evaluate model.")
    ##################
    ### DEPRECATED ###
    ##################

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return

def read_json(json_in):
    # read file
    with open(json_in, 'r') as myfile:
        data=myfile.read()

    # parse file
    obj = json.loads(data)

    return obj


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset', type=str, default='nn_trained/3by3_30deg_dat.pt',
                    help='Path to test & train datasets. Default is nn_trained')
    ap.add_argument('-w', '--weight', type=str, default='nn_trained/3by3_30deg.pth',
                    help='Path to .pth file')
    ap.add_argument('-c', '--config', type=str, default='nn_trained/3by3_30deg.json',
                     help='User configuration')
    ap.add_argument('-q', '--quant_mode', type=str, default='calib', choices=['calib', 'test'],
                    help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('-b', '--batchsize', type=int, default=64,
                    help='Testing batchsize - must be an integer. Default is 100')
    ap.add_argument('-o', '--quant_model', type=str, default='quant_model',
                    help='Path to quantize results folder. Default is quant_model')
    args = ap.parse_args()

    json_data = read_json(args.config)

    net = create_recognizer(json_data['double_frame'], json_data['net_config'])

    quantize(torch.load(args.dset), net, args.weight, json_data['M'], json_data['double_frame'],
             args.quant_mode, args.batchsize, args.quant_model, torch.device('cpu'))

    return


if __name__ == '__main__':
    run_main()
