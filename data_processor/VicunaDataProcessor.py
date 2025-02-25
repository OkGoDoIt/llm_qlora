from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from data_processor.DataProcessor import DataProcessor


class VicunaDataProcessor(DataProcessor):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_data(self) -> DatasetDict:
        if "model_context_window" in self.config:
            context_window = self.config["model_context_window"]
        else:
            context_window = self.tokenizer.model_max_length

        data = load_dataset(self.config["data"]["dataset"])
        
        data = data["train"].train_test_split(test_size=0.01)

        
        data = data.map(lambda data_point: self.tokenizer(
            self._generate_prompt(
                data_point["conversations"],
                self.tokenizer.eos_token),
            max_length=context_window,
            truncation=True,
        ))
        return data

    def _generate_prompt(self, convo: list, eos_token: str) -> str:
        convo_text = ""
        for turn in convo:
            entity = turn["from"]
            value = turn["value"]

            if entity == "human":
                convo_text += "user: "
                end_token = ""
            elif entity == "gpt":
                convo_text += "assistant: "
                end_token = eos_token  # LLM should stop its output after the response
            elif entity == "system":
                convo_text += "system: "
                end_token = ""
            else:
                print(f"WARNING: uknown entity {entity}")
                convo_text += f"### {entity.upper()}:\n"
                end_token = ""

            convo_text += value + end_token + "\n"
        return convo_text
