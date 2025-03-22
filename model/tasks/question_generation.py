import nest_asyncio
nest_asyncio.apply()

from together import AsyncTogether
import asyncio
from typing import Dict
import time

class QuestionGenerator:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.api_token = ""
        self.client = AsyncTogether(api_key=self.api_token)
        self.max_requests_per_second = 5
        self.semaphore = asyncio.Semaphore(self.max_requests_per_second)
        self.prompt = """
            Bạn được cung cấp một 'ngữ cảnh' và một 'thông tin' được lấy từ ngữ cảnh.
            Nhiệm vụ của bạn là tạo ra duy nhất một câu hỏi bằng tiếng Việt từ 'ngữ cảnh' và 'thông tin' đó.
            Câu hỏi tạo ra thỏa mãn hai yếu tố sau:
            1. Câu hỏi được tạo ra phải xuất phát từ 'ngữ cảnh'.
            2. Nếu bạn trả lời cho câu hỏi tạo ra, câu trả lời đó là 'thông tin' được cung cấp.
            Chỉ trả về kết quả là 1 câu hỏi.
        """

    async def async_chat_completion(self, requests, outputs):
        tasks = [self._rate_limited_request(messages, outputs) for messages in requests]
        await asyncio.gather(*tasks)

    async def _rate_limited_request(self, messages, outputs):
        # Use the semaphore to enforce rate limiting
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            output = response.choices[0].message.content
            output = output.strip()
            if output.startswith('"') and output.endswith('"'):
                output = output[1:-1]
            if len(output) > 0:
                outputs.append(output)

    def generate_questions(self, sample: Dict):
        input_claim = sample["input_claim"]
        claim_answer = sample["claim_answer"]

        sample["generated_question"] = []
        requests = []
        for answer in claim_answer:
            # Format input
            user_content = f"""
                Hãy tạo ra 1 câu hỏi từ ngữ cảnh và thông tin sau:
                Ngữ cảnh: {input_claim}
                Thông tin: {answer}
            """

            requests.append([
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_content},
            ])
        outputs = []

        # Use a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            coroutine = self.async_chat_completion(requests, outputs)
            task = asyncio.create_task(coroutine)
            loop.run_until_complete(task)
        else:
            asyncio.run(self.async_chat_completion(requests, outputs))

        sample["generated_question"] = outputs

        return sample


if __name__ == "__main__":
    # Input
    sample = {
        'input_claim': "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).",
        'evidence': 'SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác nêu trên. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).',
        'claim_answer': ['dự kiến từ 12 giờ ngày 25-3 ( thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)', 'công tác', 'máy nước', 'SAWACO', 'thời gian', 'bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệ p,', 'nước', 'thực hiện', 'giờ',
                        'thứ bảy)', 'thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệ p, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 ( thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)', 'bảo dưỡng định kỳ',
                        'thông báo', 'thời gian thực hiện','cung cấp','bảo trì,','4 giờ ngày 26-3 (chủ nhật)','tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệ p, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 ( thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)','dự kiến',
                        'thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến',
                        'Nhà máy nước Tân Hiệp,','ngày 26-3','công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệ p,','ngày','Tân Hiệ p,','tạm','bảo dưỡng định kỳ Nhà máy nước Tân Hiệ p,','Nhà',
                        'cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệ p, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 ( thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)',
                        '(chủ nhật)','12 giờ ngày 25-3 ( thứ bảy)','ngưng','thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệ p, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 ( thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)',
                        'Tân Hiệp']
    }

    # Question generation
    question_generator = QuestionGenerator()
    result = question_generator.generate_questions(sample)

    for i in range(len(result['claim_answer'])):
        print(f"\nAnswer: {result['claim_answer'][i]}")
        print(f"Question: {result['generated_question'][i]}")
