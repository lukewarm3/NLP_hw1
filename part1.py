import math
from collections import defaultdict
from utils import Vocab, read_data
"""You should not need any other imports."""

class NGramModel:
	def __init__(self, n, data):
		self.n = n
		self.vocab = Vocab()
		"""TODO: Populate vocabulary with all possible characters/symbols in the data, including '<BOS>', '<EOS>', and '<UNK>'."""
		self.counts = defaultdict(lambda: defaultdict(int))
		raise NotImplementedError

	def start(self):
		return ['<BOS>'] * (self.n - 1) # Remember that read_data prepends one <BOS> tag. Depending on your implementation, you may need to remove or work around that. No n-gram should have exclusively <BOS> tags; context should be n-1 <BOS> tags and the first prediction should be of the first non-BOS token.

	def fit(self, data):
		"""TODO: 
			* Train the model on the training data by populating the counts. 
				* For n>1, you will need to keep track of the context and keep updating it. 
				* Get the starting context with self.start().
		"""
		self.probs = {}
		"""TODO: Populate self.probs by converting counts to log probabilities with add-1 smoothing."""
		raise NotImplementedError

	def step(self, context):
		"""Returns the distribution over possible next symbols. For unseen contexts, backs off to unigram distribution."""
		context = self.start() + context
		context = tuple(context[-(self.n - 1):]) # cap the context at length n-1
		if context in self.probs:
			return self.probs[context]
		else:
			return {sym: math.log(1 / len(self.vocab)) for sym in self.vocab.sym2num}

	def predict(self, context):
	    """TODO: Return the most likely next symbol given a context. Hint: use step()."""
	    raise NotImplementedError

	def evaluate(self, data):
		"""TODO: Calculate and return the accuracy of predicting the next character given the original context over all sentences in the data. Remember to provide the self.start() context for n>1."""
		raise NotImplementedError

if __name__ == '__main__':

	train_data = read_data('train.txt')
	val_data = read_data('val.txt')
	test_data = read_data('test.txt')
	response_data = read_data('response.txt')

	n = 1 # TODO: n=1 and n=5
	model = NGramModel(n, train_data)
	model.fit(train_data)
	print(model.evaluate(val_data), model.evaluate(test_data))

	"""Generate the next 100 characters for the free response questions."""
	for x in response_data:
		x = x[:-1] # remove EOS
		for _ in range(100):
			y = model.predict(x)
			x += y
		print(''.join(x))
