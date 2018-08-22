import math
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.nn.init as init

from nac import NeuralAccumulatorCell
from torch.nn.parameter import Parameter

class NeuralArithmeticLogicCell(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.eps = 1e-8
		self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
		self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))
		self.nac = NeuralAccumulatorCell(in_dim, out_dim)
		self.G = Parameter(torch.Tensor(out_dim, in_dim))
		self.register_parameter('bias', None)

		init.kaiming_uniform_(self.G, a=math.sqrt(5))

	def forward(self, inputs):
		self.out_nac = self.nac(inputs)
		self.g = Func.sigmoid(Func.linear(inputs, self.G, self.bias))
		self.add_part = self.out_nac * self.g
		self.log_part = torch.log(torch.abs(inputs) + self.eps)
		self.m_part = torch.exp(self.nac(self.log_part))
		self.mul_part = (1-self.g)*self.m_part
		self.output = self.add_part + self.mul_part

		return self.output


class NALU_mutiple_cells(nn.Module):
	def __init__(self, num_layers, in_dim, hidden_dim,out_dim):
		super().__init__()
		self.num_layers = num_layers
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hidden_dim = hidden_dim

		layers = []
		for i in range(num_layers):
			layers.append(NeuralArithmeticLogicCell(
				hidden_dim if i>0 else in_dim,
				hidden_dim if i<num_layers-1 else out_dim))

		self.model = nn.Sequential(*layers)

	def forward(self, inputs):
		return self.model(inputs)
