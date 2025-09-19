import math
from collections import defaultdict
from utils import Vocab, read_data
"""You should not need any other imports."""

class NGramModel:
	def __init__(self, n, data):
		self.n = n
		self.vocab = Vocab()
		"""TODO: Populate vocabulary with all possible characters/symbols in the data, including '<BOS>', '<EOS>', and '<UNK>'."""

		for sentence in data:
			for symbol in sentence:
				self.vocab.add(symbol)
		self.vocab.add('<BOS>')
		self.vocab.add('<EOS>')
		self.vocab.add('<UNK>')
		
		self.counts = defaultdict(lambda: defaultdict(int))

	def start(self):
		return ['<BOS>'] * (self.n - 1) # Remember that read_data prepends one <BOS> tag. Depending on your implementation, you may need to remove or work around that. No n-gram should have exclusively <BOS> tags; initial context should be n-1 <BOS> tags and the first prediction should be of the first non-BOS token.

	def fit(self, data):
		"""TODO: 
			* Train the model on the training data by populating the counts. 
				* For n>1, you will need to keep track of the context and keep updating it. 
				* Get the starting context with self.start().
		"""
		context = tuple(self.start())

		for sentence in data:
			for symbol in sentence:
				if symbol not in self.vocab.sym2num:
					symbol = '<UNK>'
				
				if self.n > 1:
					# count unigrams for backoff
					self.counts[tuple()][symbol] += 1

					self.counts[context][symbol] += 1
					context = context[1:] + (symbol,)
				else:
					self.counts[context][symbol] += 1

		self.probs = defaultdict(lambda: defaultdict(float))
		"""TODO: Populate self.probs by converting counts to log probabilities with add-1 smoothing."""
		for context in self.counts:
			for symbol in self.vocab.num2sym:
				self.probs[context][symbol] = math.log(self.counts[context][symbol] + 1) - math.log(sum(self.counts[context].values()) + len(self.vocab))

	def step(self, context):
		"""Returns the distribution over possible next symbols. For unseen contexts, backs off to unigram distribution."""
		if self.n > 1:
			context = self.start() + context
			context = tuple(context[-(self.n - 1):])
		else:
			context = tuple()
			
		if context in self.probs:
			return self.probs[context]
		else:
			return self.probs[tuple()]

	def predict(self, context):
		return max(self.step(context).items(), key=lambda x: x[1])[0]
	
	def evaluate(self, data):
		total_correct = 0
		total_chars = 0
		for sentence in data:
			for i in range(len(sentence)):
				correct = sentence[i]
				context = sentence[:i]
				predicted = self.predict(context)
				if correct == predicted:
					total_correct += 1
				total_chars += 1
		return total_correct / total_chars

if __name__ == '__main__':

	train_data = read_data('data/train.txt')
	val_data = read_data('data/val.txt')
	test_data = read_data('data/test.txt')
	response_data = read_data('data/response.txt')

	for n in [1, 5]:
		print(f"n={n}")
		model = NGramModel(n, train_data)
		model.fit(train_data)
		print("accuracy on train data: ", model.evaluate(train_data))
		print("accuracy on val data: ", model.evaluate(val_data))
		print("accuracy on test data: ", model.evaluate(test_data))

		"""Generate the next 100 characters for the free response questions."""
		for x in response_data:
			x = x[:-1] # remove EOS
			for _ in range(100):
				y = model.predict(x)
				x += y
			x+="E"
			print(''.join(x))

'''
when n = 1:
accuracy on train data:  0.1729442329768594
accuracy on val data:  0.17344519423673171
accuracy on test data:  0.17463940113200657

the 100 most likely next characters for the unigram for each prompt, including the original prompt:
"<BOS>"I'm not ready to go," said                                                                                                    "
"<BOS>Lily and Max were best friends. One day                                                                                                    "
"<BOS>He picked up the juice and                                                                                                    "
"<BOS>It was raining, so                                                                                                    "
"<BOS>The end of the story was                                                                                                    "

when n = 5:
accuracy on train data:  0.6488188024892756
accuracy on val data:  0.5794273208097757
accuracy on test data:  0.5728501004199379
the 100 most likely next characters for the 5-gram for each prompt, including the original prompt:
"<BOS>"I'm not ready to go, said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the boy w"
"<BOS>Lily and Max were best friends. One day, the boy who listen the boy who listen the boy who listen the boy who listen the boy who listen the"
"<BOS>He picked up the juice and said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the "
"<BOS>It was raining, so happy and said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who l"
"<BOS>The end of the story was a little girl named lily was a little girl named lily was a little girl named lily was a little gir"

The result of the 5-gram is much better than the unigram, with 5-gram being able to generate the next character more accurately than the unigram.
For the unigram, the next character is always the same as the previous character, which has the highest probability without considering the context. 
In our case, the character with the highest probability is a blank space because it appears the most often in the training data.
For the 5-gram, the next character is more accurate than the unigram since it considers the context of the previous 4 characters. 
However, what is still lacking is that after some prediction, the 5-gram will start to repeat the same phrase over and over again.
Specifically, when the model encounters a sequence of words (a 4-gram) that it has seen many times, it will repeatedly predict the same word to complete the 5-gram.
This is caused by a lack of an understanding of the broader context of a sentence or story.
'''