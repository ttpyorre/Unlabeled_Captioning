from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

reference = [['testing', 'out', 'my', 'bleu'], ['second', 'testing', 'my', 'bleu']]
candidate = ['testing', 'my', 'bleu']
chencherry = SmoothingFunction()
score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
score2 = corpus_bleu([reference], [candidate], smoothing_function=chencherry.method1)
print(score)
print(score2)
