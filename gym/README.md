# Laser Coherent Combining Controller Environment
A custom gym environment to train reinforcement learning for laser Coherent Beam Combining(CBC) control.

## To Install Dependencies:

* To test the enviroment using  [stable baseline3](https://stable-baselines3.readthedocs.io/en/master/) algorithms install the following dependencies: 

```bash
pip3 install -r requirement.txt

```

## To train model using stable-baselines:

```bash
python rl_train_sb3.py -v -t 0.7
```

Trained model will be saved as `run/best_model.zip`.

## To log tensorboard:

* To enable tensorboard logging, you need to fill the tensorboard_log argument with a valid path, for example:

```bash
make tensorboard
```
