import numpy as np
#9行代码构建一个简单的神经网络
training_set_inputs = np.array([[0,0,1],[1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1
for iteration in range(10000):
	outputs = 1 / (1 + np.exp(-np.dot(training_set_inputs,synaptic_weights)))
	synaptic_weights += np.dot(training_set_inputs.T,(training_set_outputs - outputs)*outputs*(1 - outputs))
print(1 / (1 + np.exp( - np.dot(np.array([1,0,0]),synaptic_weights))))
