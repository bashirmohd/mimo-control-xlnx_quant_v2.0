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


def get_subgraphs():
    g = xir.Graph.deserialize('Sequential_zcu102.xmodel')
    subgraphs = get_child_subgraph_dpu(g)
    return subgraphs

def runDPU(dpu,img,verbose_io=False):

    '''get tensor'''
    outputTensors = dpu.get_output_tensors()
    output_ndim = tuple(outputTensors[0].dims)

    outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

    job_id = dpu.execute_async(img,outputData)
    dpu.wait(job_id)
    out_q = np.argmax(outputData)

    if (verbose_io is True):
        print('INPUT: ', str(img))
        print('OUTPUT: ', str(out_q))
        print('OUTPUT: ', str(outputData))


def app(model,dataIn,verbose_io=False,verbose_fps=True):

    global out_q
    out_q = [None]

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    if (verbose_fps is True):
        time1 = time.time()
        runDPU(dpu_runner, dataIn, verbose_io)
        time2 = time.time()

        timetotal = time2 - time1

        fps = float(1 / timetotal)
        print(divider)
        print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, 1, timetotal))
        print(divider)

    else:
        runDPU(dpu_runner, dataIn, verbose_io)

    return

# only used if script is run as 'main' from command line
def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()  
    ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')  
    ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',     type=str, default='Sequential_zcu102.xmodel', help='Path of xmodel. Default is Sequential_zcu102.xmodel')
    ap.add_argument('-s', '--sample',     type=str, default='sample.npy', help='Numpy array sample. Default is sample.npy')
    ap.add_argument('-v_io', '--verbose_io',   action='store_true', help='Set to print the DPU\'s input/output. Will affect the FPS performance.')
    ap.add_argument('-v_fps', '--verbose_fps',   action='store_true', help='Set to print the DPU\'s FPS performance.')

    args = ap.parse_args()  
    print(divider)
    print ('Command line options:')
    print (' --image_dir    : ', args.image_dir)
    print (' --threads      : ', args.threads)
    print (' --model        : ', args.model)
    print (' --sample       : ', args.sample)
    print (' --verbose_io   : ', args.verbose_io)
    print (' --verbose_fps  : ', args.verbose_fps)
    print(divider)

    if args.verbose_io is True:
        print('WARNING: Printing DPU\'s input/output. Will affect the FPS performance.')
  
    app(args.model,np.load(args.sample),args.verbose_io,args.verbose_fps)

    # does not work :(
    # runDPU_minimal(get_subgraphs, np.load('sample.npy'))

if __name__ == '__main__':
  main()


# does not work :(
def runDPU_minimal(subgraphs, fpgaInput):
    # vart.Runner.create_runner(subgraphs[0], "run")
    # dpu_runner = vart.Runner(subgraphs[0], "run")
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")
    fpgaOutput = [np.empty(output_ndim, dtype=np.float32, order="C")]
    jid = dpu_runner.execute_async(fpgaInput, fpgaOutput)
    dpu_runner.wait(jid)

    return