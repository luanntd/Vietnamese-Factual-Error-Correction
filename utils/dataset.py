import pandas as pd

class ViFactCheck:
    def __init__(self, base_path="hf://datasets/tranthaihoa/vifactcheck_gold_evidence/"):
        train_file="data/train-00000-of-00001.parquet"
        dev_file="data/dev-00000-of-00001.parquet"
        test_file="data/test-00000-of-00001.parquet"

        train_data = pd.read_parquet(base_path + train_file)
        dev_data = pd.read_parquet(base_path + dev_file)
        test_data = pd.read_parquet(base_path + test_file)
        all_data = pd.concat([train_data, dev_data, test_data], axis=0)

        self.processed_data = self._convert2dict(all_data)
        print("Finish loading ViFactCheck dataset")

    def _convert2dict(self, data):
        # Format and clean input
        data[['evidence', 'input_claim']] = data['input'].str.split('. Sentence: ', expand=True)
        data['evidence'] = data['evidence'].str[12:]
        data = data.drop(['input', 'instructions', 'len_evidence'], axis=1)
        data = data[data['output'] != 'Not Enough Information']
        data = data[['evidence', 'input_claim', 'output']]

        # Convert to list of dictionary
        dict_data = []
        for i in range(data.shape[0]):
            row = data.iloc[i]
            sample = {'evidence': row['evidence'], 'input_claim': row['input_claim'], 'label': row['output']}
            dict_data.append(sample)

        return dict_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_all(self):
        return self.processed_data


if __name__ == "__main__":
    # Load dataset
    data = ViFactCheck()

    # Sample view
    print(len(data))

    processed_data = data.get_all()
    print(type(processed_data))
    print(type(processed_data[0]))
    print(processed_data[:3])
