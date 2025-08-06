import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from utils import Vocab, read_data
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to use gpu on kaggle or colab

class RNNModel(nn.Module):
	def __init__(self, vocab, dims):
		super().__init__()
		self.vocab = vocab
		self.dims = dims
		"""	TODO: Initialize RNN weights/layers."""
		raise NotImplementedError

	def start(self):
		return torch.zeros(self.dims, device=device)

	def step(self, h, idx):
		"""	TODO: Pass idx through the layers of the model. Return the updated hidden state (h) and log probabilities."""
		raise NotImplementedError

	def predict(self, h, idx):
		"""	TODO: Obtain the updated hidden state and log probabilities after calling self.step(). 
			Return the updated hidden state and the most likely next symbol."""
		raise NotImplementedError

	def fit(self, data, lr=0.001, epochs=10):
		"""	TODO: Fill in the code using PyTorch functions and other functions from part2.py and utils.py.
			Most steps will only be 1 line of code. You may write it in the space below the step."""
		
		# 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
		
		# 2. Set a loss function variable to `nn.NLLLoss()` for negative log-likelihood loss.
		
		# 3. Loop through the specified number of epochs.
		
		#	 1. Put the model into training mode using `self.train()`.
		
		#	 2. Shuffle the training data using random.shuffle().
		
		#	 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of characters (`total_chars`).
		
		#	 4. Loop over each sentence in the training data.
		
		#	 	 1. Initialize the hidden state with the start state, move it to the proper device using `.to(device)`, and detach it from any previous computation graph with `.detach()`.
		
		#	 	 2. Call `optimizer.zero_grad()` to clear any accumulated gradients from the previous update.
		
		#	 	 3. Initialize a variable to keep track of the loss within a sentence (`loss`).
		
		#	 	 4. Loop through the characters of the sentence from position 1 to the end (i.e., start with the first real character, not BOS).
		
		#	 	 	1. You will need to keep track of the previous character (at position i-1) and current character (at position i). These should be expressed as numbers, not symbols.
		
		#			2. Call self.step() to get the next hidden state and log probabilities over the vocabulary given the previous character.
		
		#			3. See if this matches the actual current character (numberized). Do so by computing the loss with the nn.NLLLoss() loss initialized above. 
		#			   * The first argument is the updated log probabilities returned from self.step(). You'll need to reshape it to `(1, V)` using `.view(1, -1)`.
		#			   * The second argument is the current numberized character. It will need to be wrapped in a tensor with `device=device`. Reshape this to `(1,)` using `.view(1)`.
		
		#			4. Add this this character loss value to `loss`.

		#			5. Increment `total_chars` by 1.

		#	 	 5. After processing the full sentence, call `loss.backward()` to compute gradients.

		#		 6. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.

		#		 7. Call `optimizer.step()` to update the model parameters using the computed gradients.

		#		 8. Add `loss.item()` to `total_loss`.

		#	5. Compute the average loss per character by dividing `total_loss / total_chars`.

		#	6. For debugging, it will be helpful to print the average loss per character and the runtime after each epoch. Average loss per character should always decrease epoch to epoch and drop from about 3 to 1.2 over the 10 epochs.
		raise NotImplementedError

		def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
			Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
			Divide the total correct predictions by the total number of characters to get the final accuracy."""
		raise NotImplementedError

if __name__ == '__main__':
	
	train_data = read_data('train.txt')
	val_data = read_data('val.txt')
	test_data = read_data('test.txt')
	response_data = read_data('response.txt')

	vocab = Vocab()
	"""TODO: Populate vocabulary with all possible characters/symbols in the training data, including '<BOS>', '<EOS>', and '<UNK>'."""
	
	model = RNNModel(vocab, dims=128).to(device)
	model.fit(train_data)

	torch.save({
		'model_state_dict': model.state_dict(),
		'vocab': model.vocab,
		'dims': model.dims
	}, 'rnn_model.pth')

	"""Use this code if you saved the model and want to load it up again to evaluate. Comment out the model.fit() and torch.save() code if so.
	# checkpoint = torch.load('rnn_model.pth', map_location=device, weights_only=False)
	# vocab = checkpoint['vocab']
	# dims = checkpoint['dims']
	# model = RNNModel(vocab, dims).to(device)
	# model.load_state_dict(checkpoint['model_state_dict'])
	"""

	model.eval()

	print(model.evaluate(val_data), model.evaluate(test_data))

	"""Generate the next 100 characters for the free response questions."""
	for x in response_data:
		x = x[:-1] # remove EOS
		state = model.start()
		for char in x:
			idx = vocab.numberize(char)
			state, _ = model.step(state, idx)
		idx = vocab.numberize(x[-1])
		for _ in range(100):
			state, sym = model.predict(state, idx)
			x += sym # My predict() returns the denumberized symbol. Yours may work differently; change the code as needed.
		print(''.join(x))
