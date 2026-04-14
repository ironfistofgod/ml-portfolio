"""
Whisper large-v3 LoRA fine-tune on AMI disfluent (English).

Official Whisper fine-tuning (data collator, Trainer tokenizer, WER): https://huggingface.co/blog/fine-tune-whisper
Model card: https://huggingface.co/openai/whisper-large-v3
Dataset: https://huggingface.co/datasets/JacobLinCool/ami-disfluent
"""
import glob
import os
import shutil
import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio, load_from_disk
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

MODEL_ID   = "openai/whisper-large-v3"
LANGUAGE   = "English"
TASK       = "transcribe"
DATASET    = "JacobLinCool/ami-disfluent"
CKPT_DIR   = "/workspace/ckpts/whisper"
HF_REPO    = "chethan1988/whisper-large-v3-ami"

# Hyperparameters (override with env for sweeps / real runs)
LORA_R = int(os.environ.get("WHISPER_LORA_R", "16"))
LEARNING_RATE = float(os.environ.get("WHISPER_LR", "5e-5"))
# Smoke default 10; set WHISPER_MAX_STEPS=4000–8000+ for real training (see HF blog).
MAX_STEPS = int(os.environ.get("WHISPER_MAX_STEPS", "10"))
WARMUP_STEPS = int(os.environ.get("WHISPER_WARMUP_STEPS", "2"))
# AMI utterances can be long; 128 truncates eval. HF blog uses 225; 448 matches Whisper decode headroom.
GENERATION_MAX_LENGTH = int(os.environ.get("WHISPER_GENERATION_MAX_LENGTH", "448"))

# Persist tokenized + log-mel features on the volume so every pod start does not re-run .map().
# Default sits next to CKPT_DIR's parent so if CKPT_DIR is on your volume, cache is too.
# Override with WHISPER_PREPROCESSED_ROOT if your volume mount is not under /workspace.
# Bump WHISPER_PREPROCESS_VERSION if you change prepare_dataset() or model/dataset pairing.
PREPROCESSED_ROOT = os.environ.get(
    "WHISPER_PREPROCESSED_ROOT",
    os.path.join(os.path.dirname(CKPT_DIR), "whisper_preprocessed_cache"),
)
PREPROCESS_VERSION = os.environ.get("WHISPER_PREPROCESS_VERSION", "v1")
MAP_NUM_PROC = int(os.environ.get("WHISPER_MAP_NUM_PROC", "4"))

LORA_ALPHA     = 64
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

TRAIN_BATCH = int(os.environ.get("WHISPER_TRAIN_BATCH", "4"))
GRAD_ACCUM = int(os.environ.get("WHISPER_GRAD_ACCUM", "16"))
# For long runs set e.g. WHISPER_EVAL_STEPS=500 (blog-style). Capped to MAX_STEPS at runtime.
EVAL_STEPS = int(os.environ.get("WHISPER_EVAL_STEPS", "5"))
SAVE_STEPS = int(os.environ.get("WHISPER_SAVE_STEPS", "5"))
EVAL_BATCH = int(os.environ.get("WHISPER_EVAL_BATCH", "4"))

_hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _whisper_weights_cached(hf_home: str) -> bool:
    """Only True if snapshot dirs contain real checkpoints; avoids local_files_only with empty/partial cache."""
    root = os.path.join(hf_home, "hub", "models--openai--whisper-large-v3", "snapshots")
    if not os.path.isdir(root):
        return False
    for name in os.listdir(root):
        snap = os.path.join(root, name)
        if not os.path.isdir(snap):
            continue
        if glob.glob(os.path.join(snap, "*.safetensors")) or os.path.isfile(
            os.path.join(snap, "pytorch_model.bin")
        ):
            return True
    return False


LOCAL_ONLY = _whisper_weights_cached(_hf_home)
print(f"Whisper from_pretrained local_files_only={LOCAL_ONLY} (HF_HOME={_hf_home})")

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=LOCAL_ONLY)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK, local_files_only=LOCAL_ONLY)
processor         = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK, local_files_only=LOCAL_ONLY)

# Official Whisper fine-tuning guide loads WER once: https://huggingface.co/blog/fine-tune-whisper
wer_metric = evaluate.load("wer")

def _fs_key(s: str) -> str:
    return s.replace("/", "__")


def _preprocessed_split_dir(split: str) -> str:
    sub = f"{_fs_key(MODEL_ID)}__{_fs_key(DATASET)}__{_fs_key(LANGUAGE)}__{_fs_key(TASK)}__{PREPROCESS_VERSION}"
    return os.path.join(PREPROCESSED_ROOT, sub, split)


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def load_or_build_split(split: str):
    out_dir = os.path.abspath(_preprocessed_split_dir(split))
    info_path = os.path.join(out_dir, "dataset_info.json")
    tmp_dir = out_dir + ".__incomplete__"

    if os.path.isdir(tmp_dir):
        print(f"Removing stale incomplete preprocess dir: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if os.path.isfile(info_path):
        print(f"Loading preprocessed '{split}' from {out_dir}")
        try:
            ds = load_from_disk(out_dir)
            _ = ds[0]
            return ds
        except Exception as e:
            print(f"Cache at {out_dir} is unreadable ({e!r}). Deleting and rebuilding.")
            shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    print(f"Preprocessing '{split}' → {out_dir} (one-time; then reused from volume)")
    print(f"PREPROCESSED_ROOT (absolute) = {os.path.abspath(PREPROCESSED_ROOT)}")
    ds = load_dataset(DATASET, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.map(
        prepare_dataset,
        remove_columns=ds.column_names,
        num_proc=MAP_NUM_PROC,
    )

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    ds.save_to_disk(tmp_dir)
    if not os.path.isfile(os.path.join(tmp_dir, "dataset_info.json")):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"save_to_disk failed: missing dataset_info.json under {tmp_dir}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(tmp_dir, out_dir)
    print(f"Saved preprocessed '{split}' to {out_dir}")
    return ds


dataset_train = load_or_build_split("train")
dataset_eval = load_or_build_split("test")

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

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = np.asarray(pred.label_ids)
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    run_dir = os.path.join(CKPT_DIR, f"r{LORA_R}_lr{LEARNING_RATE}")
    print(f"Single run: LORA_R={LORA_R} LEARNING_RATE={LEARNING_RATE} MAX_STEPS={MAX_STEPS} → {run_dir}")

    eval_steps = max(1, min(EVAL_STEPS, MAX_STEPS)) if MAX_STEPS > 0 else EVAL_STEPS
    save_steps = max(1, min(SAVE_STEPS, MAX_STEPS)) if MAX_STEPS > 0 else SAVE_STEPS
    warmup_eff = min(WARMUP_STEPS, max(0, MAX_STEPS - 1))
    use_fp16 = torch.cuda.is_available()
    if not use_fp16:
        print("No CUDA: disabling fp16 (CPU/MPS debug).")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, local_files_only=LOCAL_ONLY)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    # LoRA + gradient checkpointing: without this, backward fails with
    # "element 0 of tensors does not require grad" on frozen encoder inputs.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_eff,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        push_to_hub=False,
    )

    # Official Whisper ASR guide passes feature_extractor as tokenizer:
    # https://huggingface.co/blog/fine-tune-whisper (Seq2SeqTrainer section)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"eval_wer={eval_results['eval_wer']:.4f}")

    adapter_dir = os.path.join(run_dir, "best_adapter")
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("HF_TOKEN not set, skipping upload.")
        else:
            api = HfApi(token=hf_token)
            create_repo(HF_REPO, exist_ok=True, repo_type="model", token=hf_token)

            readme_path = os.path.join(adapter_dir, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r") as f:
                    content = f.read()
                content = re.sub(r"base_model:\s*/.+", f"base_model: {MODEL_ID}", content)
                with open(readme_path, "w") as f:
                    f.write(content)

            api.upload_folder(
                folder_path=adapter_dir,
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"Adapter uploaded to {HF_REPO}")
    except Exception as e:
        print(f"HF upload failed: {e}")
