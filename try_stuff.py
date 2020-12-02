
from nltk.translate.bleu_score import sentence_bleu


reference = [['this', 'is', 'a', 'test', "a", 'b', 'c']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score)

