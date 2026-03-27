import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="metadata.csv")
    parser.add_argument("--wav_dir", type=str, default="wav")
    parser.add_argument("--max_eval_samples", type=int, default=150)
    parser.add_argument("--model_name", type=str, default="ivrit-ai/whisper-large-v3-turbo")
    parser.add_argument("--output_dir", type=str, default="./whisper-heb-ipa")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["wandb", "tensorboard"])
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)  # HuggingFace Audio column is not fork-safe with workers > 0
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()
