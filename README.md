# SimBoOpt
A simple Bayesian Optimization framework for ML and DL, which can be used for fine-tuning parameters of Models efficently. Generally, your can obtain better performance 
of your model.

## How we use it?

### Needed:
#### Define your model
do this in Model.py.
#### Setting paras of experiments
do this in template.yaml.
#### Make sure BoOp.py is modified depended on the former changes.
### If Needed:

#### Define your data function  
do this in DataProcess.py


#### Define your evaluation function
do this TrainAndEvaluate

## Examples
We have two examples of SimBoOpt: 
1. DL: CNN for classifaction of Minist 
2. ML: MAGC-based(paper: [Multi-view Attributed Graph Clustering](https://ieeexplore.ieee.org/abstract/document/9508843) for attributed graph clustering on ACM

## Dependencies
1. Numpy
2. Scipy
3. Scikit-learn
4. Ax
5. Scanpy
6. Torch
7. Traceback
8. Bayes_opt

## More details can be seen in code comment.
## Valuable contribution for this repository will be accepted.
## Cite this repository if U follow our works.

