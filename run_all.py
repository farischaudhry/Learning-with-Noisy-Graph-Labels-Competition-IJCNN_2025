import os
import subprocess

datasets = ["A", "B", "C", "D"]
os.makedirs("submission", exist_ok=True)

for dataset in datasets:
    train_path = f"data/{dataset}/train.json.gz"
    test_path = f"data/{dataset}/test.json.gz"

    print(f"\nRunning model for dataset {dataset}...\n")

    # Run main.py with WandB enabled for each dataset
    command = [
        "python", "main.py",
        "--train_path", train_path,
        "--test_path", test_path
    ]

    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = f"Dataset_{dataset}" 
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    subprocess.run(command, check=True, env=env)

print("\nAll datasets processed successfully")

# Compress the submission folder after predictions
print("\nCompressing submission folder\n")

subprocess.run([
    "python", "zipthefolder.py",
    "--input_folder", "submission",
    "--output_file", "submission.gz"
], check=True)
