import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rouge_score import rouge_scorer
import numpy as np

# Initialize variables
label_list = ["entailment", "not_entailment"]
num_labels = len(label_list)

class EntailmentModel:
    def __init__(self, entailment_model_path='vinai/phobert-base-v2', entailment_tokenizer_path="vinai/phobert-base-v2"):
        # Model and tokenizer for entailment score
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(entailment_model_path, num_labels=num_labels).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(entailment_tokenizer_path)

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_rouge(self, sentence1, sentence2):
        # Compute ROUGE scores
        scores = self.rouge_scorer.score(sentence1, sentence2)
        rouge = scores['rouge1'].fmeasure

        return rouge

    def compute_entailment(self, sample):
        sample['entailment_score'] = []
        sample['rouge_score'] = []

        sample['candidate'] = sample['candidate'] + [sample['input_claim']]  # Add input claim to handle verified claims
        for correction in sample['candidate']:
            current_entailment_scores = []

            # Compute ROUGE between input claim and candidate
            rouge_score = self.compute_rouge(correction, sample['input_claim'])
            sample['rouge_score'].append(rouge_score)

            # Handle evidence
            evidence = sample['evidence']
            encoded_evidence = self.tokenizer.encode(evidence, add_special_tokens=False)[:-1]
            encoded_correction = self.tokenizer.encode(correction, add_special_tokens=False)[1:]

            # Ensure combined length is within the maximum allowed
            max_model_length = self.tokenizer.model_max_length
            max_length = max_model_length - 3
            encoded_ctx_truncated = encoded_evidence[:max_length - len(encoded_correction)]

            # Create input_ids and attention_mask
            input_ids = [self.tokenizer.cls_token_id] + encoded_ctx_truncated + [self.tokenizer.sep_token_id] + encoded_correction + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)

            # Ensure lengths match
            # assert len(input_ids) == len(attention_mask), "Mismatch between input_ids and attention_mask lengths!"

            # Convert to tensors and move to device
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(self.device)

            # Compute entailment score
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            with torch.no_grad():
                self.model.eval()
                logits = self.model(**inputs).logits
                probs = torch.nn.Softmax(dim=1)(logits)
                entailment_prob = probs[0][0].item()  # Get entailment probability
                current_entailment_scores.append(entailment_prob)

            if len(current_entailment_scores):
                sample['entailment_score'].append(max(current_entailment_scores))

        # Combine all scores
        if sample['entailment_score']:
            sample['final_score'] = np.array(sample['entailment_score']) + np.array(sample['rouge_score']) / 50
            argmax = np.argmax(sample['final_score'])
            sample['correction'] = sample['candidate'][argmax]
        else:
            sample['correction'] = sample['input_claim']

        # Handle value type of final score:
        sample['final_score'] = sample['final_score'].tolist()
        for i in range(len(sample['final_score'])):
            sample['final_score'][i] = float(sample['final_score'][i])

        return sample


if __name__ == "__main__":
     # Input
     sample = {
         "input_claim": "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).",
         "candidate": [
             "SAWACO thông báo tạm ngưng cấp nước để thực hiện công tác bảo trì định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).",
             "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."
         ],
         "evidence": 'SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).'
     }

     # Correction scoring
     entailment_model = EntailmentModel()
     result = entailment_model.compute_entailment(sample)

     for i in range(len(result['candidate'])):
         print(f"\nCandidate: {result['candidate'][i]}")
         print(f"Entailment Score: {result['entailment_score'][i]}")
         print(f"ROUGE Score: {result['rouge_score'][i]}")
         print(f"Final Score: {result['final_score'][i]}")

     print(f"\nFinal Correction: {result['correction']}")
