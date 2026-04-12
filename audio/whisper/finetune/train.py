import os
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi, create_repo
import re
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

MODEL_ID   = "openai/whisper-large-v3"
LANGUAGE   = "English"
TASK       = "transcribe"
DATASET    = "JacobLinCool/ami-disfluent"
CKPT_DIR   = "/workspace/ckpts/whisper"
HF_REPO    = "chethan1988/whisper-large-v3-ami"

LORA_ALPHA     = 64
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

TRAIN_BATCH    = 4
GRAD_ACCUM     = 16
MAX_STEPS      = 10
WARMUP_STEPS   = 2
EVAL_STEPS     = 5
SAVE_STEPS     = 5

_hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
LOCAL_ONLY = os.path.exists(os.path.join(_hf_home, "hub", "models--openai--whisper-large-v3"))

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=LOCAL_ONLY)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK, local_files_only=LOCAL_ONLY)
processor         = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK, local_files_only=LOCAL_ONLY)

dataset_train = load_dataset(DATASET, split="train")
dataset_eval  = load_dataset(DATASET, split="test")

dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
dataset_eval  = dataset_eval.cast_column("audio",  Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

dataset_train = dataset_train.map(prepare_dataset, remove_columns=dataset_train.column_names, num_proc=4)
dataset_eval  = dataset_eval.map(prepare_dataset,  remove_columns=dataset_eval.column_names,  num_proc=4)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str   = tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
    label_str  = tokenizer.batch_decode(label_ids,  skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def train_whisper(config):
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, local_files_only=LOCAL_ONLY)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens     = []
    model.config.use_cache           = False

    lora_config = LoraConfig(
        r                = config["r"],
        lora_alpha       = LORA_ALPHA,
        target_modules   = LORA_TARGETS,
        lora_dropout     = LORA_DROPOUT,
        bias             = "none",
        use_rslora       = True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trial_dir = os.path.join(CKPT_DIR, f"r{config['r']}_lr{config['lr']}")

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = trial_dir,
        per_device_train_batch_size = TRAIN_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = config["lr"],
        warmup_steps                = WARMUP_STEPS,
        max_steps                   = MAX_STEPS,
        gradient_checkpointing      = True,
        fp16                        = True,
        evaluation_strategy         = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        logging_steps               = 25,
        predict_with_generate       = True,
        generation_max_length       = 128,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        report_to                   = "wandb" if os.environ.get("WANDB_API_KEY") else "none",
        push_to_hub                 = False,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = dataset_train,
        eval_dataset    = dataset_eval,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        tokenizer       = processor.feature_extractor,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    tune.report(wer=eval_results["eval_wer"])

    model.save_pretrained(os.path.join(trial_dir, "best_adapter"))
    print(f"Trial r={config['r']}, lr={config['lr']} → WER={eval_results['eval_wer']:.2f}")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    search_space = {
        "r":  tune.grid_search([8, 16, 32]),
        "lr": tune.grid_search([1e-5, 5e-5]),
    }

    scheduler = ASHAScheduler(
        metric="wer",
        mode="min",
        max_t=MAX_STEPS,
        grace_period=500,
    )

    analysis = tune.run(
        train_whisper,
        config=search_space,
        num_samples=1,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1},
        local_dir=os.path.join(CKPT_DIR, "ray_results"),
        name="whisper-lora-sweep",
    )

    best_config = analysis.get_best_config(metric="wer", mode="min")
    print(f"\nBest config: {best_config}")
    print(f"Best WER: {analysis.get_best_trial('wer', 'min').last_result['wer']:.2f}")

    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("HF_TOKEN not set, skipping upload.")
        else:
            api = HfApi(token=hf_token)
            create_repo(HF_REPO, exist_ok=True, repo_type="model", token=hf_token)

            best_adapter = os.path.join(
                CKPT_DIR, f"r{best_config['r']}_lr{best_config['lr']}", "best_adapter"
            )

            readme_path = os.path.join(best_adapter, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r") as f:
                    content = f.read()
                content = re.sub(r"base_model:\s*/.+", f"base_model: {MODEL_ID}", content)
                with open(readme_path, "w") as f:
                    f.write(content)

            api.upload_folder(
                folder_path=best_adapter,
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"Best adapter uploaded to {HF_REPO}")
    except Exception as e:
        print(f"HF upload failed: {e}")
