# MCLS-code
Manifold-based Constraint Laplacian Score for multi-label feature selection

This code from our paper "Manifold-based Constraint Laplacian Score for multi-label feature selection".

function idx = MCLS(fea, knear)

Inputs:

   fea: struct data set
   
       data : nSample * nFeature
       
       target : nLabel *  nSample 
       
   knear: number of neighbors

Output:

   idx: feature sorting by score
   
Usage:

fea = load('sample.mat');

knear = 5; 

idx = MCLS(fea,knear);
