# Final Project for CS536 - Machine Learning - Team 23

This repository contains the code for the final project for CS536 - Machine Learning, Spring 2022.

### Team Members:
1. Animesh Sharma
2. Karan Parsadani
3. Jaini Patel
4. Harish Udhayakumar

### Contents:
1. Base CycleGAN
2. Enhanced CycleGAN
3. Quantitative Metrics - contains scripts to compute Frechet Inception Distance and Inception Score

### Enhanced CycleGAN has the following enhancements:
- Penalize high FID between input and output for Identity Loss.
- Penalize high FID between input and output for Cyclic Consistency Loss.
- Added Global Discriminator in addition to the default PatchGAN discriminator.

### References:
1. The script to compute FID Scores relies on [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
2. The script to compute Inception Scores relies on [inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch)
