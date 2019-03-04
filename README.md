# End-to-End Efficient Representation Learning via Cascading Combinatorial Optimization
This repository has the source code for the paper "End-to-End Efficient Representation Learning via Cascading Combinatorial Optimization"(CVPR19).

## Citing this work
```
@inproceedings{jeongCVPR19,
	title= {End-to-End Efficient Representation Learning via Cascading Combinatorial Optimization},
    author={Jeong, Yeonwoo and Kim, Yoonsung and Song, Hyun Oh},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
    }
```

## Installation
* Python3.5
* Deep learning frame work : Tensorflow1.4 gpu
Check [https://github.com/tensorflow/tensorflow/tree/r1.4](https://github.com/tensorflow/tensorflow/tree/r1.4)
* Ortools(6.6.4656)
Check [https://developers.google.com/optimization/introduction/download](https://developers.google.com/optimization/introduction/download)

## Prerequisites
1. Generate directory and change path(ROOT) in **configs/path.py**
```
ROOT = '(user enter path)'
``` 
```
cd (ROOT)
mkdir exp_results
mkdir cifar_processed
```

2. Download and unzip dataset Cifar-100
```
cd RROOT/dataset
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar zxvf cifar-100-python.tar.gz
```
## Processing Data
```
cd process
python cifar_process.py
```

## Training Procedure
* In this code, we provide experiment code on 'Cifar-100'(*cifar_exps/*)
1. Metric learning model(*cifar_exps/metric/*)
    - Run *train_model* in **main.py** to train the model for specific parameter.
    - Run *integrate_results_and_preprocess* in **main.py** to integrate results and preprocess before running 'ours'.
2. Ours proposed in the paper(*cifar_exps/ours/*)
    - Run *train_model* in **main.py** to train the model for specific parameter.
    - Run *integrate_results* in **main.py** to integrate results.

## Evaluation
* Evaluation code is in **utils/evaluation.py**.
* The performance of hash table structured contructed by ours method is evaluated with 3 different metric(*NMI, precision@k, SUF*).

## License
MIT License 
