
from nltk.translate.bleu_score import sentence_bleu


reference = [['this', 'is', 'a', 'test', "a", 'b', 'c']]
candidate = ['this', 'is', 'a', 'test', "s", 'i', 'f', 'a']
score = sentence_bleu(reference, candidate)
print(score)

