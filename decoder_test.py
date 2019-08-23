import numpy as np
import pandas as pd

mean = pd.read_csv('predictions_mu.csv')
sigma = pd.read_csv('predictions_sigma.csv')
print(mean.head())
print(sigma.head())
mean_np = mean.to_numpy()
print(mean_np[0,:])

