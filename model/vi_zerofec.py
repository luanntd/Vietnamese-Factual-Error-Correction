from model.tasks.claim_answer_generation import ClaimAnswerGenerator
from model.tasks.question_generation import QuestionGenerator
from model.tasks.question_answering import QuestionAnswering
from model.tasks.qa_to_claim import QAtoClaimGenerator
from model.tasks.correction_scoring import EntailmentModel
import time
from typing import Dict, List
from tqdm import tqdm
import json

class Vi_ZeroFEC:
    def __init__(self) -> None:
        self.claim_answer_generator = ClaimAnswerGenerator()
        self.question_generator = QuestionGenerator()
        self.question_answering = QuestionAnswering()
        self.candidate_generator = QAtoClaimGenerator()
        self.score_ranking = EntailmentModel()

        print("Finish loading model.")

    def correct(self, sample: Dict):
        # Measure time for each step
        timings = {}

        # Step 1: Claim Answer Generator
        start_time = time.time()
        sample = self.claim_answer_generator.extract_information_units(sample)
        timings['ClaimAnswerGenerator'] = time.time() - start_time

        # Step 2: Question Generation
        start_time = time.time()
        sample = self.question_generator.generate_questions(sample)
        timings['QuestionGenerator'] = time.time() - start_time

        # Step 3: Question Answering
        start_time = time.time()
        sample = self.question_answering.answer_question(sample)
        timings['QuestionAnswering'] = time.time() - start_time

        # Step 4: Candidate Claim Generation
        start_time = time.time()
        sample = self.candidate_generator.generate_claims(sample)
        timings['CandidateGenerator'] = time.time() - start_time

        # Step 5: Score Ranking
        start_time = time.time()
        sample = self.score_ranking.compute_entailment(sample)
        timings['ScoreRanking'] = time.time() - start_time

        # Print timings for each step
        print("\nStep-wise Timings:")
        for step, duration in timings.items():
            print(f"{step}: {duration:.4f} seconds")

        return sample

    def batch_correct(self, samples: List[Dict], output_file: str):
        result = []
        for idx, sample in enumerate(tqdm(samples, total=len(samples))):
            try:
                # Attempt to process the sample
                processed_sample = self.correct(sample)
                result.append(processed_sample)
                with open(output_file, "a", encoding="utf-8") as f:
                    json.dump(processed_sample, f, ensure_ascii=False)
                    f.write("\n")
                    print(f"Processed and saved sample: {idx}")
            except Exception as e:
                # Log the error and continue
                print(f"Error processing sample {idx}: {e}")
                continue
        
        return result
