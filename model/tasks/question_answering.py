from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from typing import Dict
import torch

class QuestionAnswering:
    def __init__(self, model_name="PhucDanh/vit5-fine-tuning-for-question-answering"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # When use PhucDanh/vit5-fine-tuning-for-question-answering -> Add this below
        try:
            self.tokenizer.model_input_names.remove("token_type_ids")
        except:
            pass

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)

    def answer_question(self, sample: Dict):
        questions = sample['generated_question']
        sample['answer'] = []
        for question in questions:
            answer = self.pipeline(question=question, context=sample['evidence'])
            sample['answer'].append(answer['answer'])

        return sample


if __name__ == "__main__":
    # Input
    questions = {'input_claim': "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).",
                 'evidence': 'SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác nêu trên. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy), đến 4 giờ ngày 26-3 (chủ nhật).',
                 'generated_question': ['SAWACO thông báo gì?', 'Nhà máy nước Tân Hiệp đang tiến hành những công việc gì?', 'Việc ngừng cung cấp nước là tạm thời hay vĩnh viễn?', 'Dự kiến ​​lệnh đình chỉ sẽ kết thúc vào ngày nào trong tuần?', 'Ai thông báo tạm ngừng cấp nước để bảo dưỡng định kỳ Nhà máy nước Tân Hiệp?']
    }

    # Question Answering
    question_answer = QuestionAnswering()
    answers = question_answer.answer_question(questions)

    for i in range(len(answers['generated_question'])):
        print(f"\nQuestion: {answers['generated_question'][i]}")
        print(f"Answer: {answers['answer'][i]}")
