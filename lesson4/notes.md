# Lesson 4 notes

## Chapter 10 Fastbook

jargon: Self-supervised learning: Training a model using labels that are embedded in the independent variable, rather than requiring external labels. For instance, training a model to predict the next word in a text.

Tokenization:: Convert the text into a list of words (or characters, or substrings, depending on the granularity of your model)

Numericalization:: Make a list of all of the unique words that appear (the vocab), and convert each word into a number, by looking up its index in the vocab

Language model data loader creation:: fastai provides an LMDataLoader class which automatically handles creating a dependent variable that is offset from the independent variable by one token. It also handles some important details, such as how to shuffle the training data in such a way that the dependent and independent variables maintain their structure as required

Language model creation:: We need a special kind of model that does something we haven't seen before: handles input lists which could be arbitrarily big or small. There are a number of ways to do this; in this chapter we will be using a recurrent neural network (RNN). We will get to the details of these RNNs in the <>, but for now, you can think of it as just another deep neural network.

Word-based:: Split a sentence on spaces, as well as applying language-specific rules to try to separate parts of meaning even when there are no spaces (such as turning "don't" into "do n't"). Generally, punctuation marks are also split into separate tokens.
Subword based:: Split words into smaller parts, based on the most commonly occurring substrings. For instance, "occasion" might be tokenized as "o c ca sion."
Character-based:: Split a sentence into its individual characters.

jargon: token: One element of a list created by the tokenization process. It could be a word, part of a word (a subword), or a single character.

## Useful reads

[How (and why) to create a good validation set](https://www.fast.ai/posts/2017-11-13-validation-sets.html)
[The dangers of overfitting: a Kaggle postmortem](https://gregpark.io/blog/Kaggle-Psychopathy-Postmortem/)
[The problem with metrics is a big problem for AI](https://www.fast.ai/posts/2019-09-24-metrics.html)

[How to Install Without Internet](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/113195)
