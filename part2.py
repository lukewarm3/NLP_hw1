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
		self.E = nn.Embedding(len(vocab), dims)
		self.W = nn.Linear(dims, dims)
		self.U = nn.Linear(dims, dims)

		self.tanh = nn.Tanh()
		
		self.Wout = nn.Linear(dims, len(vocab))

	def start(self):
		return torch.zeros((1, self.dims), device=device)

	def step(self, h, idx): # h: size (1, dim)
		"""	TODO: Pass idx through the layers of the model. Return the updated hidden state (h) and log probabilities."""
		idx = torch.tensor(idx, device=device)
		x = self.E(idx)

		h_new = self.tanh(self.W(x) + self.U(h))
		logits = self.Wout(h_new) # (1, |V|)
		log_probs = F.log_softmax(logits, dim=1)
		return h_new, log_probs


	def predict(self, h, idx):
		"""	TODO: Obtain the updated hidden state and log probabilities after calling self.step(). 
			Return the updated hidden state and the most likely next symbol."""
		h_new, log_probs = self.step(h, idx)
		next_symbol_idx = torch.argmax(log_probs, dim=1)
		next_symbol = self.vocab.denumberize(next_symbol_idx.item())
		return h_new, next_symbol

	def fit(self, data, lr=0.001, epochs=10):
		"""	TODO: Fill in the code using PyTorch functions and other functions from part2.py and utils.py.
			Most steps will only be 1 line of code. You may write it in the space below the step."""
		
		# 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
		optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		
		# 2. Set a loss function variable to `nn.NLLLoss()` for negative log-likelihood loss.
		loss_fn = nn.NLLLoss()
		
		# 3. Loop through the specified number of epochs.
		for epoch in range(epochs):
			start_time = time.time()
		
		#	 1. Put the model into training mode using `self.train()`.
			self.train()
		
		#	 2. Shuffle the training data using random.shuffle().
			random.shuffle(data)
		
		#	 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of characters (`total_chars`).
			total_loss = 0
			total_chars = 0
		#	 4. Loop over each sentence in the training data.
			for sentence in data:
		
		#	 	 1. Initialize the hidden state with the start state, move it to the proper device using `.to(device)`, and detach it from any previous computation graph with `.detach()`.
				h = self.start().detach()
		
		#	 	 2. Call `optimizer.zero_grad()` to clear any accumulated gradients from the previous update.
				optimizer.zero_grad()
		
		#	 	 3. Initialize a variable to keep track of the loss within a sentence (`loss`).
				loss = 0
		
		#	 	 4. Loop through the characters of the sentence from position 1 to the end (i.e., start with the first real character, not BOS).
				for i in range(1, len(sentence)):
		
		#	 	 	1. You will need to keep track of the previous character (at position i-1) and current character (at position i). These should be expressed as numbers, not symbols.
					prev_char = sentence[i-1]
					curr_char = sentence[i]

					if prev_char not in self.vocab.sym2num:
						prev_char = '<UNK>'
					if curr_char not in self.vocab.sym2num:
						curr_char = '<UNK>'

					prev_idx = self.vocab.numberize(prev_char)
					curr_idx = self.vocab.numberize(curr_char)
		
		#			2. Call self.step() to get the next hidden state and log probabilities over the vocabulary given the previous character.
					h, log_probs = self.step(h, prev_idx)
		
		#			3. See if this matches the actual current character (numberized). Do so by computing the loss with the nn.NLLLoss() loss initialized above. 
		#			   * The first argument is the updated log probabilities returned from self.step(). You'll need to reshape it to `(1, V)` using `.view(1, -1)`.
		#			   * The second argument is the current numberized character. It will need to be wrapped in a tensor with `device=device`. Reshape this to `(1,)` using `.view(1)`.
					current_loss = loss_fn(log_probs.view(1, -1), torch.tensor([curr_idx], device=device).view(1))
		
		#			4. Add this this character loss value to `loss`.
					loss += current_loss

		#			5. Increment `total_chars` by 1.
					total_chars += 1

		#	 	 5. After processing the full sentence, call `loss.backward()` to compute gradients.

				loss.backward()

		#		 6. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.

				torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)

		#		 7. Call `optimizer.step()` to update the model parameters using the computed gradients.
				optimizer.step()

		#		 8. Add `loss.item()` to `total_loss`.

				total_loss += loss.item()

		#	5. Compute the average loss per character by dividing `total_loss / total_chars`.
			avg_loss = total_loss / total_chars

		#	6. For debugging, it will be helpful to print the average loss per character and the runtime after each epoch. Average loss per character should always decrease epoch to epoch and drop from about 3 to 1.2 over the 10 epochs.
			print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
			print(f"Time: {time.time() - start_time:.2f} seconds")

		
	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
			Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
			Divide the total correct predictions by the total number of characters to get the final accuracy.
		"""

		self.eval()
		with torch.no_grad():
			total_correct = 0
			total_chars = 0
			for sentence in data:
				h = self.start().to(device).detach()
				for i in range(1, len(sentence)):
					prev_char = sentence[i-1]
					curr_char = sentence[i]
					if prev_char not in self.vocab.sym2num:
						prev_char = '<UNK>'
					if curr_char not in self.vocab.sym2num:
						curr_char = '<UNK>'

					prev_idx = self.vocab.numberize(prev_char)
					h, pred_char = self.predict(h, prev_idx)
					if pred_char == curr_char:
						total_correct += 1
					total_chars += 1
		print(f"Total Correct: {total_correct}, Total Chars: {total_chars}")
		return total_correct / total_chars

if __name__ == '__main__':
	
	train_data = read_data('data/train.txt')
	val_data = read_data('data/val.txt')
	test_data = read_data('data/test.txt')
	response_data = read_data('data/response.txt')

	vocab = Vocab()
	"""TODO: Populate vocabulary with all possible characters/symbols in the training data, including '<BOS>', '<EOS>', and '<UNK>'."""
	for sentence in train_data:
		for symbol in sentence:
			vocab.add(symbol)
	vocab.add('<BOS>')
	vocab.add('<EOS>')
	vocab.add('<UNK>')

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

	print("accuracy on train data: ", model.evaluate(train_data))
	print("accuracy on val data: ", model.evaluate(val_data))
	print("accuracy on test data: ", model.evaluate(test_data))

	"""Generate the next 100 characters for the free response questions."""
	for x in response_data:
		x = x[:-1] # remove EOS
		state = model.start()
		for char in x:
			if char not in vocab.sym2num:
				char = '<UNK>'
			idx = vocab.numberize(char)
			state, _ = model.step(state, idx)

		for _ in range(100):
			idx = vocab.numberize(x[-1])
			state, sym = model.predict(state, idx)
			x += sym # My predict() returns the denumberized symbol. Yours may work differently; change the code as needed.

		print(''.join(x))

'''
Total Correct: 50272, Total Chars: 81155
accuracy on train data:  0.6194565954038568
Total Correct: 6384, Total Chars: 10766
accuracy on val data:  0.5929778933680104
Total Correct: 6343, Total Chars: 10754
accuracy on test data:  0.5898270411009857
<BOS>"I'm not ready to go," said.<EOS>™t was very saw a little girl named lily.<EOS> there was a little girl named timmy.<EOS> them to play with
<BOS>Lily and Max were best friends. One day was very saw a little girl named lily.<EOS> there was a little girl named timmy.<EOS> them to play with her
<BOS>He picked up the juice andy.<EOS> there was a little girl named timmy.<EOS> them to play with her mom said, "thank you, lily and sam a
<BOS>It was raining, son the saw a little girl named lily.<EOS> there was a little girl named timmy.<EOS> them to play with her mom
<BOS>The end of the story was and said, "thank you, lily and sam and said, "thank you, lily and sam and said, "thank you, lily an
'''
