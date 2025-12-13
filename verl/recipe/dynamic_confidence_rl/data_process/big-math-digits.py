from datasets import load_dataset
from system_prompts import INSTRUCT_PROMPT

if __name__ == "__main__":
    # train data
    train_data = load_dataset("mehuldamani/big-math-digits", split="train")

    def process_fn_train(example, idx):
        data = {
            "data_source": "big-math-digits",
            "prompt": [
                {"role": "user", "content": INSTRUCT_PROMPT + example["problem"]},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "index": idx,
                "llama8b_solve_rate": example["llama8b_solve_rate"],
            },
        }
        return data

    train_dataset = train_data.map(
        function=process_fn_train,
        with_indices=True,
        remove_columns=train_data.column_names,
    )
    train_dataset.to_parquet("./big_math_digits.parquet")
    print(f"Train: {len(train_dataset)}")
