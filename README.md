# NLP HW1

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.0).

## Part 1
* Unigram accuracy: training: 0.1729442329768594, validation: 0.17344519423673171, testing: 0.17463940113200657
* 5-gram accuracy: training: 0.6488188024892756, validation: 0.5794273208097757, testing: 0.5728501004199379

Free response: 

**the 100 most likely next characters for the unigram for each prompt, including the original prompt:**

"<BOS>"I'm not ready to go," said                                                                                                    "

"<BOS>Lily and Max were best friends. One day                                                                                                    "

"<BOS>He picked up the juice and                                                                                                    "

"<BOS>It was raining, so                                                                                                    "

"<BOS>The end of the story was     

**the 100 most likely next characters for the 5-gram for each prompt, including the original prompt:**

"<BOS>"I'm not ready to go," said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the boy w"

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

## Part 2
* RNN accuracy: training: 0.6194565954038568, validation: 0.5929778933680104, testing: 0.5898270411009857
* Link to saved model: `rnn_model.pth` file in the current path

## Part 3
* LSTM accuracy: training: 0.6595773519807775, validation: 0.6175924205833179, testing: 0.614190068811605
* Link to saved model: `lstm_model.pth` file in the current path

Free response:

output for RNN:

<BOS>"I'm not ready to go," said.<EOS>™t was very saw a little girl named lily.<EOS> there was a little girl named timmy.<EOS> them to play with

<BOS>Lily and Max were best friends. One day was very saw a little girl named lily.<EOS> there was a little girl named timmy.<EOS> them to play with her

<BOS>He picked up the juice andy.<EOS> there was a little girl named timmy.<EOS> them to play with her mom said, "thank you, lily and sam a

<BOS>It was raining, son the saw a little girl named lily.<EOS> there was a little girl named timmy.<EOS> them to play with her mom

<BOS>The end of the story was and said, "thank you, lily and sam and said, "thank you, lily and sam and said, "thank you, lily an


output for LSTM:

<BOS>"I'm not ready to go," saidy the sun and the sun and the sun and the sun and the sun and the sun and the sun and the sun and th

<BOS>Lily and Max were best friends. One day, they were happy.<EOS> the boy named timmy.<EOS> they were happy.<EOS> the boy named timmy.<EOS> they were happy.<EOS> 

<BOS>He picked up the juice and the sun and the sun and the sun and the sun and the sun and the sun and the sun and the sun and the

<BOS>It was raining, son.<EOS> lily saw a big made a big made a sun.<EOS>" lily saw a big made a big made a sun.<EOS>" lily saw a big m

<BOS>The end of the story was a little girl named lily.<EOS> lily saw a big made a big made a big made a sun.<EOS>" lily saw a big made a


Question a. How does coherence compare between the vanilla RNN and LSTM? 

Answer a. LSTM model works better since it catches more context in the previous sentence. For example, LSTM generates "Lily and Max were best friends. One day, they were happy." while for the same example RNN generates "Lily and Max were best friends. One day was very saw a little girl named lily." which is not logical.

Question b. Concretely, how do the neural methods compare with the n-gram models?

Answer b. The neural methods (RNN and LSTM) demonstrate a clear advantage over the n-gram models, particularly in terms of generating more coherent and contextually relevant text. The output of n-gram is highly repetitive and lack meaning, while neural models, though imperfect, shows some contextual awareness. Specifically, while the RNN outputs are still repetitive ("there was a little girl named timmy"), they show a better understanding of sentence structure and can introduce new characters. LSTM has a better performance that it is less prone to repetitiveness.

Question c. What is still lacking? What could help make these models better?

Answer c. What is still lacking is the ability to write correct grammar and syntax, have better logical coherence, and the awareness of longer context. To make the model better, we can use larger database so that the model can have more knowledge of the language. We can also try to use more advanced model like Transformer.

