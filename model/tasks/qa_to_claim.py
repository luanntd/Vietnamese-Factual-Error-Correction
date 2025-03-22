import nest_asyncio
nest_asyncio.apply()

from together import AsyncTogether
import asyncio
from typing import Dict
import time

class QAtoClaimGenerator:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.api_token = ""
        self.client = AsyncTogether(api_key=self.api_token)
        self.max_requests_per_second = 5
        self.semaphore = asyncio.Semaphore(self.max_requests_per_second)
        self.prompt = f"""
                Nhiệm vụ của bạn là tạo ra một câu tuyên bố từ cặp câu hỏi và câu trả lời cho trước. Câu tuyên bố phải mang đầy đủ nội dung của cả câu hỏi và câu trả lời.
                Bằng cách kết hợp câu hỏi và câu trả lời, hãy tạo ra một câu tuyên bố duy nhất.
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

    def generate_claims(self, sample: Dict):
        questions = sample["generated_question"]
        answers = sample["answer"]

        sample["candidate"] = []
        requests = []
        for question, answer in zip(questions, answers):
            # Format input
            user_content = f"""
                    Câu hỏi: {question}
                    Trả lời: {answer}
                    Câu tuyên bố:
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

        sample["candidate"] = outputs

        return sample


if __name__ == "__main__":
    # Input sample
    sample = {
        "input_claim": "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp.",
        "answer": [
            'SAWACO',
            'tạm',
            '26-3 (chủ nhật).',
            "từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."
        ],
        "generated_question": [
            'Ai thông báo tạm ngừng cấp nước để bảo dưỡng định kỳ Nhà máy nước Tân Hiệp?',
            'Việc ngừng cung cấp nước là tạm thời hay vĩnh viễn?',
            'Dự kiến ​lệnh đình chỉ sẽ kết thúc vào ngày nào trong tuần?',
            'Việc cung cấp nước sẽ bị ngừng trong bao nhiêu ngày?'
        ]
    }

    # QA-to-Claim
    qa_to_claim_generator = QAtoClaimGenerator()
    result = qa_to_claim_generator.generate_claims(sample)

    for i in range(len(result["generated_question"])):
        print(f"\nQuestion: {result['generated_question'][i]}")
        print(f"Answer: {result['answer'][i]}")
        print(f"Candidate: {result['candidate'][i]}")
