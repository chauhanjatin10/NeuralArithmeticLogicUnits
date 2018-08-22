import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as Func

class NeuralAccumulatorCell(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim

		self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
		self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))
		self.W = Parameter(Func.tanh(self.W_hat) * Func.sigmoid(self.M_hat))
		self.register_parameter('bias', None)

		init.kaiming_uniform_(self.W_hat, a=math.sqrt(5))
		init.kaiming_uniform_(self.M_hat, a=math.sqrt(5))

	def forward(self, inputs):
		return Func.linear(inputs, self.W, self.bias)


class NAC_multiple_cells(nn.Module):
	def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
		super().__init__()
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.out_dim = out_dim
		self.num_layers = num_layers

		layers = []

		for i in range(num_layers):
			layers.append(NeuralAccumulatorCell(
				hidden_dim if i>0 else in_dim,
				hidden_dim if i < num_layers-1 else out_dim))

		self.model = nn.Sequential(*layers)

	def forward(self, inputs):
		return self.model(inputs)


class NeuralAccumulatorCell_Recurrent(nn.Module):
	def __init__(self, in_dim, out_dim, hidden_dim):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hidden_dim = hidden_dim
		self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
		self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))
		self.W = Parameter(Func.tanh(self.W_hat) * Func.sigmoid(self.M_hat))
		self.W_hat_hidden = Parameter(torch.Tensor(hidden_dim, in_dim))
		self.M_hat_hidden = Parameter(torch.Tensor(hidden_dim, in_dim))
		self.W_hidden = Parameter(Func.tanh(self.W_hat_hidden) * Func.sigmoid(self.M_hat_hidden))

		self.register_parameter('bias', None)

		init.kaiming_uniform_(self.W_hat, a=math.sqrt(5))
		init.kaiming_uniform_(self.M_hat, a=math.sqrt(5))
		init.kaiming_uniform_(self.W_hat_hidden, a=math.sqrt(5))
		init.kaiming_uniform_(self.M_hat_hidden, a=math.sqrt(5))

	def forward(self, inputs, hidden_prev):
		return Func.tanh(Func.linear(inputs, self.W, self.bias)+Func.linear(hidden_prev, self.W_hidden, self.bias))


class Recurrent_NAC(nn.Module):
	def __init__(self, num_steps, input_dim, output_dim, hidden_dim):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_steps = num_steps
		self.hidden_dim = hidden_dim
		self.hidden0 = torch.FloatTensor(self.hidden_dim).uniform_(*[0,1]).unsqueeze_(1)


	def forward(self, x):
		self.nac = NeuralAccumulatorCell_Recurrent(x.shape, self.hidden_dim, self.hidden_dim)
		self.hidden = self.hidden0
		#self.hidden_layer_output_dict = {}
		for i in range(self.num_steps-1):
			self.hidden = self.nac(x[i], self.hidden)
		self.nac_output = NeuralAccumulatorCell_Recurrent(x.shape, self.output_dim, self.hidden_dim)
		output = self.nac_output(x[num_steps-1], self.hidden)
		return output
