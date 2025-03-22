from typing import Dict
import stanza
import torch

# Function to get all sentence phrases
def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = []
    for child in tree.children:
        results += get_phrases(child, label)

    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results

class ClaimAnswerGenerator:
    def __init__(self):
        self.nlp_stanza = stanza.Pipeline(
            lang="vi",
            processors="tokenize,ner,pos,constituency",
            use_gpu=torch.cuda.is_available()
        )

    def extract_information_units(self, sample: Dict):
        # Use Stanza to annotate the text
        doc = self.nlp_stanza(sample['input_claim'])

        # Extract entities
        ents = []
        spe_char = [".", ",", "(", ")", "!"]
        for ent in doc.ents:
            if ent.text not in spe_char:
                ents.append(ent.text)

        # Extract word types (POS tagging)
        ents += [
            word.text
            for sentence in doc.sentences
            for word in sentence.words
            if word.upos in ['VERB', 'NOUN', 'ADJ', 'ADV']
        ]

        # Extract noun phrases and verb phrases
        ents += [phrase for sent in doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
        ents += [phrase for sent in doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]

        # Extract negation terms
        negations = [word for word in ["không", "chưa", "chẳng"] if word in sample['input_claim']]

        # Extract the middle of the sentence: the main content of the sentence
        middle = []
        start_match = ""
        end_match = ""
        for ent in ents:
            if sample['input_claim'].startswith(ent) and len(ent) > len(start_match):
                start_match = ent
            if sample['input_claim'].endswith(ent + ".") and len(ent) > len(end_match):
                end_match = ent

        if len(start_match) > 0 and len(end_match) > 0:
            middle.append(sample['input_claim'][len(start_match):-len(end_match) - 1].strip())

        # Combine all extractions
        results = list(set(ents + negations + middle))
        for i in range(len(results)):
            results[i] = results[i].replace("_", " ")
        sample['claim_answer'] = results

        return sample


if __name__ == "__main__":
    # Input sample
    sample = {
        'input_claim': "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).",
        'evidence': 'SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác nêu trên. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).'
    }

    # Claim answer generation
    generator = ClaimAnswerGenerator()
    sample = generator.extract_information_units(sample)
    print(f"\nInformation unit: {sample['claim_answer']}")
    print(f"Number of infor unit: {len(sample['claim_answer'])}")
