# CartPole with Random Gravity,Friction and Noise

This is our submission for the major ML project. 

## Requirements

The dependencies for this project are included in the mlproject.yml file.

Use the following command to install the environment

```
conda env create -f environment.yml
```

## Files Included

A brief explanation of all files in this directory.

        images:- Contains training log images for dqn model

        models:- Contains trained models.

                t1model.h5:- Uncompressed model for task1
        
                t2model.h5:- Uncompressed model for task2

                t3model.h5:- Uncompressed model for task3

                cartpole-dqnt1short.h5:- Compressed model for task1

                cartpole-dqnt2short.h5:- Compressed model for task2

                cartpole-dqnt3short.h5:- Compressed model for task3

        finaltask.py:- Model loader and tester for all 3 tasks. Note- We are only using cartpole-dqnt1short.h5 as final submission for all 3 tests.

        traint1t2.py:- Training code for getting cartpole-dqnt1short.h5. Note:- the same code was used for other models with more or less changes.

        task1.py:- CartPole but with Random Gravity and Friction

        task2.py:- Random Gravity and Friction along with noisy controls

        task3.py:- Everything as previous along with noisy sensors

## Authors

* **Anupreet Singh**
* **Urvil Jivani**
* **Adithya Samavedhi**