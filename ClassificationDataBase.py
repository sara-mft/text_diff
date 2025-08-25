from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet as wn

def multilingual_meteor_score(references, hypothesis, lang="eng"):
    """
    METEOR score using multilingual WordNet (OMW).
    Args:
        references (list[str]): reference translations
        hypothesis (str): hypothesis translation
        lang (str): ISO language code ('eng','fra','spa',...)
    """
    def expand(tokens):
        expanded = set(tokens)
        for tok in tokens:
            for syn in wn.synsets(tok, lang=lang):
                for lemma in syn.lemmas(lang):
                    expanded.add(lemma.name().replace("_", " "))
        return list(expanded)

    # Expand tokens for hypothesis and references
    hyp_tokens = hypothesis.split()
    hyp_expanded = expand(hyp_tokens)

    ref_expanded = [expand(ref.split()) for ref in references]

    # meteor_score expects tokenized inputs
    return meteor_score(ref_expanded, hyp_expanded)



def calculate_meteor(self, lang="eng") -> float:
    scores = [
        multilingual_meteor_score(refs, hyp, lang=lang)
        for refs, hyp in zip(self.references, self.hypotheses)
    ]
    return sum(scores) / len(scores)
