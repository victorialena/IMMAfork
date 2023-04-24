# (Fork of) Interaction Modeling with Multiplex Attention
Authors: [Fan-Yun Sun](https://cs.stanford.edu/~sunfanyun/), [Isaac Kauvar](https://ikauvar.github.io/), [Ruohan Zhang](https://ai.stanford.edu/~zharu/), [Jiachen Li](https://jiachenli94.github.io/), [Mykel Kochenderfer](https://mykel.kochenderfer.com/), [Jiajun Wu](https://jiajunwu.com/), [Nick Haber](https://ed.stanford.edu/faculty/nhaber)

This repository contains the abbrivaited version of the above paper ([arxiv](https://arxiv.org/abs/2208.10660), [openreview](https://openreview.net/forum?id=SeHslYhFx5-)).
The original repo can be found [here](https://github.com/sunfanyunn/IMMA).

## Environment Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
3. Install necessary packages with pip
```
pip install -r requirements.txt
```

## Data Setup
Download the preprocessed dataset [here](https://drive.google.com/drive/folders/14bxUhp2K4BZYFk9d3wQkWj9BLbY36VAy?usp=sharing) (or run `gdown 'https://drive.google.com/drive/folders/14bxUhp2K4BZYFk9d3wQkWj9BLbY36VAy?usp=sharing'`) and place it under `datasets`. 


## Run code
```
run_bball.py
run_spring.py
```