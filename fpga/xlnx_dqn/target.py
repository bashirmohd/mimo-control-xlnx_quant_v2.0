'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Make the target folder
Copies images, application code and compiled xmodel to 'target'
'''

'''
Author: Mark Harvey
'''

import torch
import torchvision

import argparse
import os
import shutil
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'feedback'))
from train_nn import CombiningDataGen
from recognizer import create_recognizer

from torch.utils.data import Dataset, DataLoader, random_split


DIVIDER = '-----------------------------------------'

# def generate_images(dset_dir, num_images, dest_dir):

#     # cb: not MNIST anymore  
#     # classes = ['zero','one','two','three','four','five','six','seven','eight','nine']

#     # # MNIST test dataset and dataloader
#     # test_dataset = torchvision.datasets.MNIST(dset_dir,
#     #                                         train=False, 
#     #                                         download=True,
#     #                                         transform=gen_transform)

#     # test_loader = torch.utils.data.DataLoader(test_dataset,
#     #                                         batch_size=1, 
#     #                                         shuffle=True)

#     # hardcode for now
#     config = {
#         'n_samples': 5000,
#         'M': 3,
#         'test_3_in_9': False
#     }

#     dataset = CombiningDataGen(**config)
#     vali_num = int(0.1 * len(dataset))
#     train_num = len(dataset) - vali_num
#     print('vali_num =', vali_num)
#     print('train_num =', train_num)
#     train_dataset, vali_dataset = random_split(dataset, [train_num, vali_num])

#     # cb: batchsize = 1 questionable
#     batchsize = 1
#     test_loader = torch.utils.data.DataLoader(train_dataset,
#                                               batch_size=batchsize,
#                                               shuffle=True)



#     # iterate thru' the dataset and create images
#     dataiter = iter(test_loader)
#     for i in tqdm(range(num_images)):
#         image, label = dataiter.next()
#         img = image.numpy().squeeze()
#         img = (img * 255.).astype(np.uint8)
#         idx = label.numpy()
#         img_file=os.path.join(dest_dir, classes[idx[0]]+'_'+str(i)+'.png')
#         cv2.imwrite(img_file, img)

#     return


def make_target(target, this_path):

    app_path = this_path + '/application'
    target_path = this_path + '/target_' + target
    model_path = this_path + '/compiled_model/Sequential_' + target + '.xmodel'

    # remove any previous data
    shutil.rmtree(target_path, ignore_errors=True)    
    os.makedirs(target_path)

    # copy application code
    print('Copying application code from',app_path,'...')
    shutil.copy(os.path.join(app_path, 'app_test.py'), target_path)

    print('Copying application code from',app_path,'...')
    shutil.copy(os.path.join(app_path, 'app_minimal.py'), target_path)

    # copy compiled model
    print('Copying compiled model from',model_path,'...')
    shutil.copy(model_path, target_path)

    print('Done copying to', target_path, '!')

    # cb: do not really have to generate MNIST images.
    # create images
    # dest_dir = target_dir + '/images'
    # shutil.rmtree(dest_dir, ignore_errors=True)  
    # os.makedirs(dest_dir)
    # generate_images(dset_dir, num_images, dest_dir)


    return



def main():

    this_path = os.path.dirname(os.path.realpath(__file__))
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--target', type=str,  default='zcu102', choices=['zcu102','u50','vck190'], help='Target board type (zcu102,u50,vck190). Default is zcu102')
    args = ap.parse_args()  

    # print('\n------------------------------------')
    # print(sys.version)
    # print('------------------------------------')
    # print ('Command line options:')
    # print (' --dset_dir     : ', args.dset_dir)
    # print (' --target       : ', args.target)
    # print (' --num_images   : ', args.num_images)
    # print (' --app_dir      : ', args.app_dir)
    # print('------------------------------------\n')


    make_target(args.target, this_path)


if __name__ ==  "__main__":
    main()
