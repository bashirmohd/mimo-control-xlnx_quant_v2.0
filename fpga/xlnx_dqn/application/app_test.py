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

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse


divider='---------------------------'

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(dpu,img,verbose=False):

    '''get tensor'''
    outputTensors = dpu.get_output_tensors()
    output_ndim = tuple(outputTensors[0].dims)

    outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

    job_id = dpu.execute_async(img,outputData)
    dpu.wait(job_id)
    out_q = np.argmax(outputData)

    if (verbose is True):
        print('INPUT: ', str(img))
        print('OUTPUT: ', str(out_q))
        print('OUTPUT: ', str(outputData))


def app(threads,model,dataIn,verbose=False):

    global out_q
    out_q = [None]

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []    
    in_q = dataIn
    t1 = threading.Thread(target=runDPU, args=(all_dpu_runners[i], in_q, verbose))
    threadAll.append(t1)
        
    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(1 / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, 1, timetotal))
    print(divider)

    return




# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='CNN_zcu102.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
  ap.add_argument('-v', '--verbose',   action='store_true', help='Set to print the DPU\'s input/output. Will affect the FPS performance.')

  args = ap.parse_args()  
  print(divider)
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)
  print (' --verbose     : ', args.verbose)
  print(divider)

  if args.verbose is True:
    print('WARNING: Printing DPU\'s input/output. Will affect the FPS performance.')
  
  app(args.threads,args.model,np.load('sample.npy'),args.verbose)

if __name__ == '__main__':
  main()

