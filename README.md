# Deep Reinforcement Learning for Financial Portfolio Optimisation
TD3-Based Ensemble Reinforcement Learning for Financial Portfolio Optimisation. Created for masters dissertation, Department of Artificial Intelligence, Faculty of Information & Communication Technology University of Malta.
Nigel Cuschieri, supervised by Vince Vella and Josef Bajada.

![Ensemble](https://user-images.githubusercontent.com/15673499/119581482-ec423880-bdc2-11eb-8628-2a0d21b6edcd.png)

## Installation instructions
Our project was created on Ubuntu 20, using Anaconda. PIP and conda requirements are available in both .txt and .yml format. You may choose one as you deem fit. These are: 
* **requirements.txt** - ```pip install --user --requirement requirements.txt```
* **tensor_keras_portfolio.yml** - ```conda env create -f tensor_keras_portfolio.yml``` 

## Datasets
Datasets are in H5 format, stored in _utils/datasets_.
* **NYSE(N)** dataset, which may be found here found [here](http://www.mysmu.edu.sg/faculty/chhoi/olps/datasets/NYSE_N_2_Dataset.html)
* **SP500** custom dataset based on S&P500, with real stock data gathered using Yahoo Finance.
New datasets can be created using the jupyter notebooks found in _utils/_.!

## Settings
Main configuration is done in _config/stock.json_.

## Training
### Pre Trained models
The trained models used in our study are saved within the solution. The training results are saved in _results/nyse_n_ and _results/SP500_. The weights are stored in _weights/nyse_n_ and _weights/SP500_.
### Training process
* Open cmd in the folder, and connect to the environment (_conda activate tensor_keras_portfolio_).
* Execute the **stock_trading.py** python file. Note that this will overwrite the corresponding training results and weights. Parameters include:
    * Debug (_-d = True_)
    * Framework (_-f = DDPG_)
    * Predictor Type (_-p = LSTM_)
    * Window Length (_-w = 3_)
    * Technical Indicators (_-t = False_)

## Testing
Multiple Jupyter notebooks are included in our project to allow for testing and evaluation of our trained models. These include:
* **Finding Threshold Value for Training TD3 models.ipynb** - Used to study the hypothetical rewards given to OLPS algorithms to justify the creation of a _Value Function Threshold_.
* **On-line Portfolio Selection.ipynb** - Tests on solely OLPS algorithms.
* **Test DDPG and TD3.ipynb** - Tests to compare models with the two different frameworks.
* **TD3 vs TD3-rmr-pred.ipynb** - Tests to compare performance of enhanced state format.
* **Test DDPG and TD3-InDepth.ipynb** - In depth analysis of performance is done with additional visualisations allowing us to see decisions made by agent.
* **Test Ensemble.ipynb** - Tests to compare performance of the ensemble model with OLPS algorithms and previous models.
* **Training Results.ipynb** - Visualisations of training results.

![Plot](https://user-images.githubusercontent.com/15673499/119581530-09770700-bdc3-11eb-9712-93b667f351c0.png)

## References
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem (Jiang et. al.)](https://arxiv.org/abs/1706.10059)
* [Continuous control with deep reinforcement learning (Hunt et. al.)](https://arxiv.org/abs/1509.02971)
* [Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et. al.)](https://arxiv.org/abs/1802.09477), [Code on Github](https://github.com/sfujim/TD3)
* The code is inspired by [Using Reinforcement Learning for Portfolio Optimization](https://github.com/bassemfg/ddpg-rl-portfolio-management) 
* The environment is inspired by [wassname/rl-portfolio-management](https://github.com/wassname/rl-portfolio-management)
* DDPG implementation is inspired by [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
* OLPS algorithms borrowed and extended from [universal-portfolios](https://github.com/Marigold/universal-portfolios)
