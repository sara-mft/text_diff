from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# Example: source sentences (references) and target sentences (hypotheses)
references = [
    ["the cat is on the mat".split()],
    ["there is a cat on the mat".split()]
]

hypotheses = [
    "the cat is on the mat".split(),
    "a cat is on the mat".split()
]

# Corpus BLEU score
bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
print("Corpus BLEU Score:", bleu_score)

# Sentence-level BLEU scores
for ref, hyp in zip(references, hypotheses):
    score = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
    print("Sentence BLEU:", score)
