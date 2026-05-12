import json 
import yaml 
import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert SQL assistant.
Generate only SQL queries.
"""

#load config 
with open("configs/trainings.yml", "r") as f:
    config = yaml.safe_load(f)

MAX_SAMPLES = config['dataset']['max_samples']

train_data = []
def prepare_sft_dataset():
    try:
        dataset = load_dataset("b-mc2/sql-create-context")
        for i, sample in enumerate(dataset['train']):
            if i>= MAX_SAMPLES:
                break

            user_prompt = f"""
    Schema:
    {sample['context']}

    Question:
    {sample['question']}
    """
            assistant_response = sample['answer']
            formatted_sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            train_data.append(formatted_sample)

    except Exception as e:
        logger.error(f"Error preparing SFT dataset: {e}")

    with open("datasets/sft/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    logger.info(f"SFT dataset prepared with {len(train_data)} samples and saved to datasets/sft/train.json")


if __name__ == "__main__":
    prepare_sft_dataset()



