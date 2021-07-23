# Deep Learning based control for Lasers and Accelerators
PIs: Qiang Du (ALS), Mariam Kiran (ESnet)

## How to run on Lawrencium

### Setup access to gitlab.lbl.gov

Modify `~/.ssh/config` as following, makesure `~/.ssh/id_rsa` is one of your enabled ssh keys:

```
Host gitlab.lbl.gov
   IdentityFile ~/.ssh/id_rsa

Host *
   IdentityFile ~/.ssh/perceus
   StrictHostKeyChecking=no
```

### Load modules for tensorflow/2.1.0 with py3.7

Add the following line to `~/.bashrc`:

```bash
module load ml/tensorflow/2.1.0-py37
```

### Setup Jupyter server:

Setup new kernel for `ml/tensorflow/2.1.0-py37` by putting the following in `$HOME/.ipython/kernels/mykernel/kernel.json`:

```json
{
    "argv": [
        "/global/software/sl-7.x86_64/modules/langs/python/3.7/bin/python3",
        "-m",
        "IPython.kernel",
        "-f",
        "{connection_file}"
    ],
    "display_name": "Python 3.7",
    "language": "python",
    "env": {
        "PATH": "/global/software/sl-7.x86_64/modules/apps/ml/tensorflow/2.1.0-py37/bin:/global/software/sl-7.x86_64/modules/langs/cuda/10.1/bin:/global/software/sl-7.x86_64/modules/langs/python/3.7/bin:/global/software/sl-7.x86_64/modules/tools/vim/7.4/bin:/global/software/sl-7.x86_64/modules/tools/emacs/25.1/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/global/home/groups/allhands/bin",
        "PYTHONPATH": "/global/software/sl-7.x86_64/modules/apps/ml/tensorflow/2.1.0-py37/lib/python3.7/site-packages"
    }
}
```

### Spawn server and run

Follow section 1 in [instructions](https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/getting-started/jupyter-notebook)

### To Run interactive job on Lawrencium with GPU support:

```bash
srun  -N 1 -p es1 -A pc_dl4acc -t 1:0:0 --gres=gpu:4 -n 8 -q es_normal --pty bash
```

### To Run Xilinx Quantization:
```
./docker_run.sh xilinx/vitis-ai-cpu:latest
```
or:
```
./docker_run.sh xilinx/vitis-ai-cpu
```
and then do:
```
source run_quant.sh
```
from within the `quantization` directory