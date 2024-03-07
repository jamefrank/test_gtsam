# -*- encoding: utf-8 -*-
#@File    :   main.py
#@Time    :   2024/03/07 16:50:08
#@Author  :   frank 
#@Email:
#@Description: gtsam自定义因子拟合曲线

import gtsam
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from functools import partial

# 生成含有噪声的数据
gt_param = np.array([1.0,2.0,1.0])
w_sigma = 1.0;  
x_data = np.array([i/100.0 for i in range(100)])

def funct(p, x):
    return np.exp(p[0] * x * x + p[1] * x + p[2])

rng = np.random.default_rng(12345)
y_data = [funct(gt_param, xi) + rng.normal(scale=w_sigma) for xi in x_data]

def error_func(xi:float, yi:float, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]) -> np.ndarray:
    k0 = this.keys()[0]
    k1 = this.keys()[1]
    k2 = this.keys()[2]
    # print(k0,k1,k2)
    a = v.atVector(k0)
    b = v.atVector(k1)
    c = v.atVector(k2)
    # print(a,b,c)
    val = funct([a,b,c], xi)
    if H is not None:
        H[0] = xi*xi*val
        H[1] = xi*val
        H[2] = val

    error = val-yi
    return error

#
graph = gtsam.NonlinearFactorGraph()

noiseM = gtsam.noiseModel.Diagonal.Sigmas([w_sigma])
for xi,yi in zip(x_data,y_data):
    factor = gtsam.CustomFactor(noiseM, gtsam.KeyVector([0,1,2]), partial(error_func,xi,yi))
    graph.add(factor)

initial = gtsam.Values()
initial.insert(0, np.array([2.0]))
initial.insert(1, np.array([-1.0]))
initial.insert(2, np.array([5.0]))

#
params = gtsam.GaussNewtonParams()
optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
result = optimizer.optimize()

print(result)

# plt.figure(figsize= (6.4, 4.8))
# plt.plot(x_data, y_data, "ro")
# # plt.plot(x, y, {{"color", "blue"}, {"label", "$y = e^{ax^2+bx+c}$"}})
# plt.show()