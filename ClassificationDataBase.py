from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet as wn

def multilingual_meteor_score(references, hypothesis, lang="eng"):
    """
    METEOR score using multilingual WordNet (OMW), preserving word order.
    Args:
        references (list[str]): reference sentences
        hypothesis (str): hypothesis sentence
        lang (str): ISO language code ('eng','fra','spa',...)
    """

    def expand_token(token, lang):
        # Keep the original token
        synonyms = {token}
        for syn in wn.synsets(token, lang=lang):
            for lemma in syn.lemmas(lang):
                synonyms.add(lemma.name().replace("_", " "))
        return list(synonyms)

    def expand_sentence(sentence, lang):
        tokens = sentence.split()
        return [expand_token(tok, lang) for tok in tokens]

    # Expand hypothesis and references while preserving order
    hyp_expanded = expand_sentence(hypothesis, lang)
    refs_expanded = [expand_sentence(ref, lang) for ref in references]

    # meteor_score expects: references: list of list-of-tokens, hypothesis: list-of-tokens
    # But since we now have alternatives, we need to pick the base tokens
    # Simple trick: take the first synonym in each list (original token first)
    def pick_original(expanded_tokens):
        return [alts[0] for alts in expanded_tokens]

    hyp_final = pick_original(hyp_expanded)
    refs_final = [pick_original(ref) for ref in refs_expanded]

    return meteor_score(refs_final, hyp_final)
