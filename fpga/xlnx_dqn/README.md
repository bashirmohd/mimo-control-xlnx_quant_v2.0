To start, run:
```
$ ./docker_run.sh xilinx/vitis-ai-cpu:latest
```
or:
```
$ ./docker_run.sh xilinx/vitis-ai-cpu
```

You will then be navigated back to the root directory of the repo.

From within the docker container, run:
```
> conda activate vitis-ai-pytorch
```

Then, navigate into the `fpga/xlnx_dqn` directory and run the following command to perform quantization with training:
```
$ ./run_quant.sh --train
```

If training has already been performed, you may omit the `--train` flag. Be advised that it is important to run with the `--train` flag first from within the docker. That is, if training had already been performed _outside_ the docker, the `--train` flag _must_ be used, to avoid `pytorch` incompatibilites between Xilinx's `pytorch` version and whatever `pytorch` version was used to train the model natively by the user beforehand.
