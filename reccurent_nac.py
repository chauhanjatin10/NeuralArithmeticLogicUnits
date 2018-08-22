import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as Func
import torch.nn.init as init
import numpy as np
from nac import NeuralAccumulatorCell_Recurrent as na_cell


class Recurrent_NAC(nn.Module):
	def __init__(self, num_steps, input_dim, output_dim, hidden_dim):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_steps = num_steps
		self.hidden_dim = hidden_dim
		self.hidden0 = torch.FloatTensor(self.hidden_dim).uniform_(*[0,1]).unsqueeze_(1)


	def forward(self, x):
		self.nac = na_cell(x.shape, self.hidden_dim, self.hidden_dim)
		self.hidden = self.hidden0
		#self.hidden_layer_output_dict = {}
		for i in range(self.num_steps-1):
			self.hidden = self.nac(x[i], self.hidden)
		self.nac_output = na_cell(x.shape, self.output_dim, self.hidden_dim)
		output = self.nac_output(x[num_steps-1], self.hidden)
		return output


