import pandas as pd
import numpy as np
import operator

pred1 = pd.read_csv('out/MyLgbModel-4.csv')
pred2 = pd.read_csv('out/MyLstm-None.csv')
pred3 = pd.read_csv('out1/MyLgbModel-wmd.csv')

pred1['prediction'] = pred1['prediction'] * 0.9 + pred2['prediction'] * 0.05 + pred3['prediction'] * 0.05

pred1.to_csv("out/ens.csv", index = False)