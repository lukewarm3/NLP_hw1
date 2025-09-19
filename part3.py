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
		self.E = nn.Embedding(len(vocab), dims)
		self.Wout = nn.Linear(dims, len(vocab))

		self.Wf = nn.Linear(dims, dims)
		self.Uf = nn.Linear(dims, dims)

		self.Wi = nn.Linear(dims, dims)
		self.Ui = nn.Linear(dims, dims)

		self.Wo = nn.Linear(dims, dims)
		self.Uo = nn.Linear(dims, dims)

		# candidate cell
		self.Wg = nn.Linear(dims, dims)
		self.Ug = nn.Linear(dims, dims)

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

	def start(self):
		h = torch.zeros((1, self.dims), device=device)
		c = torch.zeros((1, self.dims), device=device)
		return (h, c)

	def step(self, state, idx):
		"""	TODO: Pass idx through the layers of the model. 
			Return a tuple containing the updated hidden state (h) and cell state (c), and the log probabilities of the predicted next character."""
		idx = torch.tensor(idx, device=device)
		x = self.E(idx)

		h, c = state

		f = self.sigmoid(self.Wf(x) + self.Uf(h))
		i = self.sigmoid(self.Wi(x) + self.Ui(h))
		o = self.sigmoid(self.Wo(x) + self.Uo(h))
		c_in = self.tanh(self.Wg(x) + self.Ug(h))

		c_new = f * c + i * c_in
		h_new = o * self.tanh(c_new)

		logits = self.Wout(h_new)
		log_probs = F.log_softmax(logits, dim=1)
		return (h_new, c_new), log_probs
		

	def predict(self, state, idx):
		"""	TODO: Obtain the updated state and log probabilities after calling self.step(). 
			Return the updated state and the most likely next symbol."""
		state_new, log_probs = self.step(state, idx)
		next_symbol_idx = torch.argmax(log_probs, dim=1)
		next_symbol = self.vocab.denumberize(next_symbol_idx.item())
		return state_new, next_symbol

	def fit(self, data, lr=0.001, epochs=10):
		"""	TODO: This function is identical to fit() from part2.py. 
			The only exception: the state to keep track is now the tuple (h, c) rather than just h. This means after initializing the state with the start state, detatch it from the previous computattion graph like this: `(state[0].detach(), state[1].detach())`"""
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
				state = self.start()
				state[0].detach()
				state[1].detach()
		
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
					state, log_probs = self.step(state, prev_idx)
		
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
			The code may be identitcal to evaluate() from part2.py."""
		self.eval()
		with torch.no_grad():
			total_correct = 0
			total_chars = 0
			for sentence in data:
				state = self.start()
				state[0].detach()
				state[1].detach()
				for i in range(1, len(sentence)):
					prev_char = sentence[i-1]
					curr_char = sentence[i]
					if prev_char not in self.vocab.sym2num:
						prev_char = '<UNK>'
					if curr_char not in self.vocab.sym2num:
						curr_char = '<UNK>'

					prev_idx = self.vocab.numberize(prev_char)
					state, pred_char = self.predict(state, prev_idx)
					if pred_char == curr_char:
						total_correct += 1
					total_chars += 1
		print(f"Total Correct: {total_correct}, Total Chars: {total_chars}")
		return total_correct / total_chars

if __name__ == '__main__':
	
	vocab = Vocab()
	vocab.add('<BOS>')
	vocab.add('<EOS>')
	vocab.add('<UNK>')

	train_data = read_data('data/train.txt')
	val_data = read_data('data/val.txt')
	test_data = read_data('data/test.txt')
	response_data = read_data('data/response.txt')

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

	print("accuracy on train data: ", model.evaluate(train_data))
	print("accuracy on val data: ", model.evaluate(val_data))
	print("accuracy on test data: ", model.evaluate(test_data))

	for x in response_data:
		x = x[:-1] # remove EOS
		state = model.start()
		for char in x:
			if char not in vocab.sym2num:
				char = '<UNK>'
			idx = vocab.numberize(char)
			state, _ = model.predict(state, idx)
		
		for _ in range(100):
			idx = vocab.numberize(x[-1])
			state, sym = model.predict(state, idx)
			x += sym # My predict() returns the denumberized symbol. Yours may work differently; change the code as needed.
		print(''.join(x))
'''
Total Correct: 53528, Total Chars: 81155
evaluation on train data:  0.6595773519807775
Total Correct: 6649, Total Chars: 10766
evaluation on val data:  0.6175924205833179
Total Correct: 6605, Total Chars: 10754
evaluation on test data:  0.614190068811605
<BOS>"I'm not ready to go," saidy the sun and the sun and the sun and the sun and the sun and the sun and the sun and the sun and th
<BOS>Lily and Max were best friends. One day, they were happy.<EOS> the boy named timmy.<EOS> they were happy.<EOS> the boy named timmy.<EOS> they were happy.<EOS> 
<BOS>He picked up the juice and the sun and the sun and the sun and the sun and the sun and the sun and the sun and the sun and the
<BOS>It was raining, son.<EOS> lily saw a big made a big made a sun.<EOS>" lily saw a big made a big made a sun.<EOS>" lily saw a big m
<BOS>The end of the story was a little girl named lily.<EOS> lily saw a big made a big made a big made a sun.<EOS>" lily saw a big made a
'''