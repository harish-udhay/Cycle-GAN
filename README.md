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

![image](https://user-images.githubusercontent.com/70934463/181502438-bfa01839-a012-4493-9e89-0c7848f7809c.png)
![image](https://user-images.githubusercontent.com/70934463/181502485-5eda9106-b8fd-435f-9416-5461f96566ea.png)
![image](https://user-images.githubusercontent.com/70934463/181502531-6b215f25-7242-4a3c-b847-a95d0d036a29.png)
![image](https://user-images.githubusercontent.com/70934463/181502579-66f4fb50-dcd3-4ac0-aeef-3d382804f3bf.png)
![image](https://user-images.githubusercontent.com/70934463/181502613-39e35ad0-1fea-430a-a327-42e9eba12b83.png)
![image](https://user-images.githubusercontent.com/70934463/181502649-ba53bdc9-3e8f-44be-a1a0-f2911786cb88.png)
![image](https://user-images.githubusercontent.com/70934463/181502679-96140715-df07-45cc-a8d6-0e423b633c1f.png)
![image](https://user-images.githubusercontent.com/70934463/181502719-7c1a80a3-50cf-41e4-a84e-98366b80cff9.png)
