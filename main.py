from utils.dataset import ViFactCheck
from model.vi_zerofec import Vi_ZeroFEC

# Load model
corrector = Vi_ZeroFEC()

# Load all dataset
vifactcheck = ViFactCheck("hf://datasets/tranthaihoa/vifactcheck_gold_evidence/")
dataset = vifactcheck.get_all()

# Correct samples
samples = dataset[0:500]
outputs = corrector.batch_correct(samples, "/content/drive/MyDrive/NLP/outputs.jsonl")
