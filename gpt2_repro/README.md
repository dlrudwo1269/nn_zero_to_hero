# Train GPT-2
## Contents
This directory contains the following files/directories:
1. `train_gpt2.ipynb`: The Jupyter Notebook that summarizes the video lecture [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=9f7SBliNbRyM66dP)
2. `train_gpt2.py`: Training script to replicate GPT training
3. `gpt2_modules.py`: A module that defines the modules used in the GPT architecture used in `train_gpt2.py`
4. `dataloaderlite.py`: A module that defines the dataloader for the [FineWeb-Edu](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) dataset to use for training (tokenize and shard) dataset, supporting DDP
5. `fineweb.py`: A Python script to prepare the [FineWeb-Edu](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) dataset to use for training (tokenize and shard)
6. `log.txt`: Contains sample logs from training (for the first 200 steps)

## Training Details
A sample training run was performed on a Lambda Labs `gpu_8x_a100` instance for 1 hour to verify that training works as expected. It took roughly 30 minutes to preprocess the training data using `fineweb.py`. The output from the first few iterations of the training loop are shown below:
```
total desired batch size: 524288
=> calculated gradient accumulation steps: 4
found 99 shards for split train
found 1 shards for split val
num decayed parameter tensors: 50, with 124,354,560 parameters
num non-decayed parameter tensors: 98, with 121,344 parameters
using fused AdamW: True
Validation loss 10.9535
step    0 | loss: 10.955009 | lr: 1.2000e-05 | norm: 15.9685 | dt: 12701.84 | tok/sec: 41276.54
step    1 | loss: 10.440608 | lr: 2.4000e-05 | norm: 9.0330 | dt: 498.35 | tok/sec: 1052056.01
step    2 | loss: 10.040066 | lr: 3.6000e-05 | norm: 5.4874 | dt: 493.48 | tok/sec: 1062435.81
step    3 | loss: 9.791483 | lr: 4.8000e-05 | norm: 3.3288 | dt: 493.22 | tok/sec: 1062981.73
step    4 | loss: 9.677402 | lr: 6.0000e-05 | norm: 2.8487 | dt: 493.14 | tok/sec: 1063154.92
step    5 | loss: 9.606796 | lr: 7.2000e-05 | norm: 2.5278 | dt: 493.42 | tok/sec: 1062568.77
step    6 | loss: 9.557916 | lr: 8.4000e-05 | norm: 2.4614 | dt: 492.88 | tok/sec: 1063721.13
step    7 | loss: 9.504156 | lr: 9.6000e-05 | norm: 2.3396 | dt: 492.42 | tok/sec: 1064723.90
step    8 | loss: 9.411872 | lr: 1.0800e-04 | norm: 2.2048 | dt: 492.35 | tok/sec: 1064874.97
step    9 | loss: 9.331245 | lr: 1.2000e-04 | norm: 2.0591 | dt: 492.35 | tok/sec: 1064872.39
step   10 | loss: 9.214384 | lr: 1.3200e-04 | norm: 2.0187 | dt: 492.70 | tok/sec: 1064117.48
step   11 | loss: 9.125767 | lr: 1.4400e-04 | norm: 1.9298 | dt: 492.83 | tok/sec: 1063822.51
step   12 | loss: 9.022243 | lr: 1.5600e-04 | norm: 1.9116 | dt: 494.21 | tok/sec: 1060868.95
step   13 | loss: 8.922011 | lr: 1.6800e-04 | norm: 1.7625 | dt: 493.41 | tok/sec: 1062579.04
step   14 | loss: 8.754240 | lr: 1.8000e-04 | norm: 1.8547 | dt: 493.06 | tok/sec: 1063326.62
step   15 | loss: 8.654723 | lr: 1.9200e-04 | norm: 1.7356 | dt: 493.22 | tok/sec: 1062990.98
step   16 | loss: 8.554069 | lr: 2.0400e-04 | norm: 1.5308 | dt: 493.35 | tok/sec: 1062709.47
step   17 | loss: 8.408154 | lr: 2.1600e-04 | norm: 1.4749 | dt: 493.46 | tok/sec: 1062467.63
step   18 | loss: 8.413359 | lr: 2.2800e-04 | norm: 1.3324 | dt: 492.79 | tok/sec: 1063915.67
step   19 | loss: 8.350785 | lr: 2.4000e-04 | norm: 1.5422 | dt: 493.23 | tok/sec: 1062960.15
step   20 | loss: 8.069128 | lr: 2.5200e-04 | norm: 1.2495 | dt: 493.60 | tok/sec: 1062171.01
step   21 | loss: 7.961398 | lr: 2.6400e-04 | norm: 1.1039 | dt: 493.74 | tok/sec: 1061865.83
step   22 | loss: 7.844496 | lr: 2.7600e-04 | norm: 0.8049 | dt: 493.34 | tok/sec: 1062721.28
step   23 | loss: 7.805030 | lr: 2.8800e-04 | norm: 0.8709 | dt: 494.39 | tok/sec: 1060475.52
step   24 | loss: 7.638826 | lr: 3.0000e-04 | norm: 1.4175 | dt: 494.07 | tok/sec: 1061156.65
step   25 | loss: 7.539103 | lr: 3.1200e-04 | norm: 0.7864 | dt: 494.17 | tok/sec: 1060941.11
step   26 | loss: 7.548504 | lr: 3.2400e-04 | norm: 0.5878 | dt: 494.19 | tok/sec: 1060898.63
step   27 | loss: 7.451670 | lr: 3.3600e-04 | norm: 0.6764 | dt: 495.00 | tok/sec: 1059158.74
step   28 | loss: 7.417964 | lr: 3.4800e-04 | norm: 0.6636 | dt: 494.30 | tok/sec: 1060672.45
step   29 | loss: 7.341280 | lr: 3.6000e-04 | norm: 1.0638 | dt: 494.51 | tok/sec: 1060215.28
```

We see that the loss and gradient norm are decreasing as we expect and the learning rate warmup is taking place. We are processing ~1000000 tokens per second with DDP and the various other optimizations we made in the code.

Some sample output at step 100 (which is mainly gibberish, but just making sure the code is reasonable):
```
rank 1 sample3: Hello, I'm a language model, if you need more than far as they made that do but don't understand a lot they really like to have any kind
rank 0 sample0: Hello, I'm a language model, and, then, it has not been, for that means, if, so, but are, the idea of it
rank 0 sample1: Hello, I'm a language model, who found that was not be in course, the right if it will cause a few, to the past of the last
rank 0 sample2: Hello, I'm a language model, it is there we might then for example, a little. This will be a set to be a regular. The value
rank 0 sample3: Hello, I'm a language model, we found that it is, it must not you, so that this means to allow this week, to be better deal
rank 2 sample0: Hello, I'm a language model, one day. And you've a bit more than that it.
How you.
You can be better. It
rank 2 sample1: Hello, I'm a language model, who I also I just the words not very important; I got it you. I had the classroom and you will get
rank 2 sample2: Hello, I'm a language model, I. You may be a word
you don't find: What do, the same night to do. Once:
rank 2 sample3: Hello, I'm a language model, is, my students to speak, and their story who, as well as. He is probably far in her to see
rank 3 sample0: Hello, I'm a language model, and the world, you know what their name. They do things. Don't be found what does going to a simple
rank 3 sample1: Hello, I'm a language model, who had the world's who would have a very as being the same thing which is the place the truth at one thing
rank 3 sample2: Hello, I'm a language model, and a small time so this week, to give it has gone in a bit things, the teacher, or as to
rank 3 sample3: Hello, I'm a language model, who so that I cannot always trying to be not the child, which may be really need about in other
Here's
```