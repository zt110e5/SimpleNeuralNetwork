import numpy as np
'''
a multi-layered neural network in Python
'''
class NeuronLayer():
	def __init__(self,numbers_of_neurons,numbers_of_input_per_neuron,):
		self.synaptic_weights = 2 * np.random.random((numbers_of_input_per_neuron,numbers_of_neurons)) - 1

class NeuralNetwork():
	def __init__(self,layer1,layer2):
		self.layer1 = layer1
		self.layer2 = layer2

	def sigmoid(self,x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self,x):
		return x * (1 - x)

	# We train the neural network through a process of trial and error.
	# Adjusting the synaptic weights each time.
	def train(self,train_set_inputs,train_set_outputs,numbers_of_train_iteration,):
		for iteration in range(numbers_of_train_iteration):
			outputs_from_layer1,outputs_from_layer2 = self.think(train_set_inputs)

			layer2_error = train_set_outputs - outputs_from_layer2
			layer2_delta = layer2_error * self.sigmoid_derivative(outputs_from_layer2)

			layer1_error = np.dot(layer2_delta,self.layer2.synaptic_weights.T)
			layer1_delta = layer1_error * self.sigmoid_derivative(outputs_from_layer1)

			layer1_adjustment = np.dot(train_set_inputs.T,layer1_delta)
			layer2_adjustment = np.dot(outputs_from_layer1.T,layer2_delta)

			self.layer2.synaptic_weights += layer2_adjustment
			self.layer1.synaptic_weights += layer1_adjustment


	#caculate the output
	def think(self,inputs):
		outputs_from_layer1 = self.sigmoid(np.dot(inputs,self.layer1.synaptic_weights))
		outputs_from_layer2 = self.sigmoid(np.dot(outputs_from_layer1,self.layer2.synaptic_weights))
		return outputs_from_layer1,outputs_from_layer2

	# The neural network prints its weights
	def print_weights(self):
		print("    Layer 1 (4 neurons, each with 3 inputs): ")
		print(self.layer1.synaptic_weights)
		print("    Layer 2 (1 neurons, each with 4 inputs): ")
		print(self.layer2.synaptic_weights)


if __name__ == '__main__':
	# Seed the random number generator
	np.random.seed(1)

	# Create layer 1 (4 neurons, each with 3 inputs)
	# Create layer 2 (a single neuron with 4 inputs)
	layer1 = NeuronLayer(4,3)
	layer2 = NeuronLayer(1,4)

	# Combine the layers to create a neural network
	neural_network = NeuralNetwork(layer1,layer2)

	print("Stage 1) Random starting synaptic weights: ")
	neural_network.print_weights()

	training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
	training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

	neural_network.train(training_set_inputs,training_set_outputs,50000)

	print("Stage 2) New synaptic weights after training: ")
	neural_network.print_weights()

	# Test the neural network
	print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
	hidden_state, output = neural_network.think(np.array([1, 1, 0]))
	print(output)
