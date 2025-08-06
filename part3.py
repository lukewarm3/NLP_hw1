import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import Vocab, read_data
import time
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
	def __init__(self, vocab, dims):
		super().__init__()
		self.vocab = vocab
		self.dims = dims
		"""	TODO: Initialize LSTM weights/layers."""
		raise NotImplementedError

	def start(self):
		h = torch.zeros(self.dims, device=device)
		c = torch.zeros(self.dims, device=device)
		return (h, c)

	def step(self, state, idx):
		"""	TODO: Pass idx through the layers of the model. 
			Return a tuple containing the updated hidden state (h) and cell state (c), and the log probabilities of the predicted next character."""
		raise NotImplementedError

	def predict(self, state, idx):
		"""	TODO: Obtain the updated state and log probabilities after calling self.step(). 
			Return the updated state and the most likely next symbol."""
		raise NotImplementedError

	def fit(self, data, lr=0.001, epochs=10):
		"""	TODO: This function is identical to fit() from part2.py. 
			The only exception: the state to keep track is now the tuple (h, c) rather than just h. This means after initializing the state with the start state, detatch it from the previous computattion graph like this: `(state[0].detach(), state[1].detach())`"""
		raise NotImplementedError

	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
			Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
			Divide the total correct predictions by the total number of characters to get the final accuracy.
			The code may be identitcal to evaluate() from part2.py."""
		raise NotImplementedError

if __name__ == '__main__':
	
	vocab = Vocab()
	vocab.add('<BOS>')
	vocab.add('<EOS>')
	vocab.add('<UNK>')

	train_data = read_data('train.txt')
	val_data = read_data('val.txt')
	test_data = read_data('test.txt')
	response_data = read_data('response.txt')

	for sent in train_data:
		vocab.update(sent)
	model = LSTMModel(vocab, dims=128).to(device)
	model.fit(train_data)
	torch.save({
		'model_state_dict': model.state_dict(),
		'vocab': model.vocab,
		'dims': model.dims
	}, 'lstm_model.pth')

	"""Use this code if you saved the model and want to load it up again to evaluate. Comment out the model.fit() and torch.save() code if so.
	# checkpoint = torch.load('rnn_model.pth', map_location=device, weights_only=False)
	# vocab = checkpoint['vocab']
	# dims = checkpoint['dims']
	# model = RNNModel(vocab, dims).to(device)
	# model.load_state_dict(checkpoint['model_state_dict'])
	"""
	
	model.eval()

	print(model.evaluate(val_data), model.evaluate(test_data))

	for x in response_data:
		x = x[:-1] # remove EOS
		state = model.start()
		for char in x:
			idx = vocab.numberize(char)
			state, _ = model.predict(state, idx)
		idx = vocab.numberize(x[-1])
		for _ in range(100):
			state, idx = model.predict(state, idx)
			x += sym # My predict() returns the denumberized symbol. Yours may work differently; change the code as needed.
		print(''.join(x))
