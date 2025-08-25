from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet as wn

def multilingual_meteor_score(references, hypothesis, lang="eng"):
    """
    METEOR score using multilingual WordNet (OMW).
    Args:
        references (list[str]): list of reference sentences
        hypothesis (str): hypothesis sentence
        lang (str): language code ('eng', 'fra', 'spa', etc.)
    """
    def expand_with_synonyms(tokens):
        expanded = set(tokens)
        for tok in tokens:
            for syn in wn.synsets(tok, lang=lang):
                for lemma in syn.lemmas(lang):
                    expanded.add(lemma.name().replace("_", " "))
        return list(expanded)

    # Expand synonyms in both refs and hyp
    expanded_refs = [" ".join(expand_with_synonyms(ref.split())) for ref in references]
    expanded_hyp = " ".join(expand_with_synonyms(hypothesis.split()))

    return meteor_score(expanded_refs, expanded_hyp)



def calculate_meteor(self, lang="eng") -> float:
    scores = [
        multilingual_meteor_score(refs, hyp, lang=lang)
        for refs, hyp in zip(self.references, self.hypotheses)
    ]
    return sum(scores) / len(scores)
