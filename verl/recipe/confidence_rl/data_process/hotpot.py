from datasets import DatasetDict, load_dataset
from system_prompts import get_sys_prompt

HOTPOT_PROMPT = get_sys_prompt("instruct")
DATA_SOURCE = "hotpot_qa"


def _build_prompt(problem_text: str) -> list[dict]:
    """Embed the system prompt directly into the user message (same pattern as big-math-digits)."""
    return [{"role": "user", "content": HOTPOT_PROMPT + problem_text}]


def _format_example(example, idx):
    return {
        "data_source": DATA_SOURCE,
        "prompt": _build_prompt(example["problem"]),
        "ability": "hotpot",
        "reward_model": {"style": "rule", "ground_truth": example["answer"]},
        "extra_info": {
            "index": idx,
            "id": example.get("id"),
            "source": example.get("source"),
            "type": example.get("type"),
            "level": example.get("level"),
            "gold_removed": example.get("gold_removed"),
            "removed_titles": example.get("removed_titles"),
        },
    }


def _process_split(dataset, split_name: str, output_path: str):
    processed = dataset.map(
        function=_format_example,
        with_indices=True,
        remove_columns=dataset.column_names,
    )
    processed.to_parquet(output_path)
    print(f"{split_name.capitalize()} split: {len(processed)} samples -> {output_path}")
    return processed


if __name__ == "__main__":
    train_data = load_dataset("mehuldamani/hotpot_qa", split="train")
    test_data = load_dataset("mehuldamani/hotpot_qa", split="test")

    processed_train = _process_split(train_data, "train", "./hotpot_train.parquet")
    processed_test = _process_split(test_data, "test", "./hotpot_test.parquet")

    dataset_dict = DatasetDict({"train": processed_train, "test": processed_test})
    dataset_dict.save_to_disk("./hotpot_hf_dataset")
    print("Saved Hugging Face dataset to ./hotpot_hf_dataset")
