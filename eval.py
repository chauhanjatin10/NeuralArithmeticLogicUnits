import torch
import math
import torch.nn.functional as Func
import torch.nn as nn
import numpy as np
import random
from nalu import NALU_mutiple_cells
from nac import NAC_multiple_cells, Recurrent_NAC
#from reccurent_nac import Recurrent_NAC

def generate_data_interpolation(arithmetic_type, num_vals_for_sum, num_train, num_test, input_range, support):
	data = torch.FloatTensor(input_range).uniform_(*support).unsqueeze_(1)
	X, y = [], []
	for i in range(num_train + num_test):
		idx_a = random.sample(range(input_range), num_vals_for_sum)
		idx_b = random.sample([x for x in range(input_range) if x not in idx_a], num_vals_for_sum)
		a, b = data[idx_a].sum(), data[idx_b].sum()
		X.append([a, b])
		y.append(arithmetic_type(a, b))
	X = torch.FloatTensor(X)
	y = torch.FloatTensor(y).unsqueeze_(1)
	indices = list(range(num_train + num_test))
	np.random.shuffle(indices)
	X_train, y_train = X[indices[num_test:]], y[indices[num_test:]]
	X_test, y_test = X[indices[:num_test]], y[indices[:num_test]]
	return X_train, y_train, X_test, y_test
   

def generate_data_extrapolation(arithmetic_type, num_examples, input_range, support=[50, 100]):
	data = torch.FloatTensor(input_range).uniform_(*support).unsqueeze_(1)
	X, y = [], []
	for i in range(num_examples):
		index_a = random.sample(range(input_range), 10)
		index_b = random.sample([x for x in range(input_range) if x not in index_a] , 10)
		a, b = data[index_a].sum(), data[index_b].sum()
		X.append([a, b])
		y.append(arithmetic_type(a, b))
	X = torch.FloatTensor(X)
	y = torch.FloatTensor(y)
	return X, y


def generate_data_mnist():
	mnist = np.load('/home/jatin/codes/deeplearning/digits_training/mnist_scaled.npz')
	X_train = mnist['X_train'][:1000]
	y_train = mnist['y_train'][:1000]
	X_test = mnist['X_test'][1000:1100]
	y_test = mnist['y_test'][1000:1100]
	#print(np.unique(y_train))
	mean_vals = np.mean(X_train,axis=0)
	std_val = np.std(X_train)
	X_train_centered = (X_train - mean_vals)/std_val
	X_test_centered = (X_test - mean_vals)/std_val

	X_train = torch.from_numpy(X_train_centered)
	X_test = torch.from_numpy(X_test_centered)
	y_train = torch.from_numpy(y_train)
	y_test = torch.from_numpy(y_test)
	return X_train[:5000], y_train[:5000], X_test[5000:6000], y_test[5000:6000]


def train(model, X_train, y_train, optimizer):
	for i in range(1000):
		output = model(X_train)
		loss = Func.mse_loss(output, y_train)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i%1000 == 0:
			print("\t{}/{}: loss: {:.7f}".format(
                i+1, 10000, loss.item()))

def test(model, X_test, y_test):
	with torch.no_grad():
		out = model(X_test)
		return torch.abs(out - y_test)

def eval_function_arithmetic():
	arithmetic_type = {
	'add': lambda x, y: x+y,
	'subtract': lambda x, y: x-y,
	'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x, y: torch.pow(x, 2),
    'root': lambda x, y: torch.sqrt(x),
	}

	results_arithmetic_functions = {}

	for func_name, func in arithmetic_type.items():
		results_arithmetic_functions[func_name] = []

		X_train, y_train, X_test, y_test = generate_data_interpolation(
            num_train=500, num_test=50,
            input_range=100, num_vals_for_sum=5, arithmetic_type=func,
            support=[5, 10],
        )
		X_extrapolated, y_extrapolated = generate_data_extrapolation(func, 100, 1000)
		
		model = NALU_mutiple_cells(
            num_layers=2,
            in_dim=2, hidden_dim = 3, out_dim=1)
		optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
		train(model, X_train, y_train, optimizer)
		y_pred = test(model, X_test, y_test)
		mean_absolute_error = test(model, X_test, y_test).mean().item()
		results_arithmetic_functions[func_name].append(mean_absolute_error)
		#print(y_test, y_pred)
		
	#print(results_arithmetic_functions)
	'''with open("interpolation.txt", "w") as f:
        f.write("NALU\n")
        for k, v in results_arithmetic_functions.items():
            rand = results_arithmetic_functions[k][0]
            mean_absolute_errors = [100.0*x/rand for x in results_arithmetic_functions[k][1:]]
            if NORMALIZE:
                f.write("{:.3f}\n".format(mean_absolute_errors))
            else:
                f.write("{:.3f}\n".format(results_arithmetic_functions[k][1:]))'''


def eval_mnist():
	X_train, y_train, X_test, y_test = generate_data_mnist()
	model = Recurrent_NAC(10, 784, 1, 500)
	optimizer = torch.optim.RMSprop(list(model.parameters()), lr=1e-3)
	for i in range(500):
		output = model(X_train[i*10:(i+1)*10])
		loss = Func.mse_loss(output, torch.sum(y_train[i*10:(i+1)*10]))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("Training step {}".format(i))

	Y_pred = model(X_test)
	loss = Func.mse_loss(output, y_test)
	print(y_test, Y_pred)

eval_mnist()