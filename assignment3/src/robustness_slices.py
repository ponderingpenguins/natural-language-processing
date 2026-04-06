"""Run robustness slices on AG News for BERT/LSTM/CNN."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluation import load_model
from utils.dataset import dataset_prep

SRC_DIR = Path(__file__).resolve().parent
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if os.getenv("REQUIRE_CUDA", "0").lower() in {"1", "true", "yes"} and DEVICE.type != "cuda":
    raise RuntimeError(
        "REQUIRE_CUDA is enabled but CUDA is unavailable. Aborting to prevent CPU fallback."
    )

OUT_DIR = SRC_DIR / "robustness_slices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROOT_DIR = SRC_DIR.parents[1]
A2_SRC_DIR = ROOT_DIR / "assignment2" / "src"
A2_CNN_ARTIFACT_DIR = SRC_DIR / "a2_cnn_slice_model"
A2_CNN_MODEL_PATH = A2_CNN_ARTIFACT_DIR / "model.pt"
A2_CNN_TOKENIZER_PATH = A2_CNN_ARTIFACT_DIR / "tokenizer.pkl"
A2_CNN_META_PATH = A2_CNN_ARTIFACT_DIR / "meta.json"

MODEL_SPECS = {
    "bert": {
        "model_path": Path(
            os.getenv(
                "ROBUST_BERT_MODEL_PATH",
                str(SRC_DIR / "bert_output/sanity_distilbert/checkpoint-20250"),
            )
        ),
        "config_path": Path(
            os.getenv(
                "ROBUST_BERT_CONFIG_PATH",
                str(SRC_DIR / "bert_output/sanity_distilbert/config.json"),
            )
        ),
    },
    "lstm": {
        "model_path": Path(
            os.getenv(
                "ROBUST_LSTM_MODEL_PATH",
                str(SRC_DIR / "experiment_outputs/updated_default_full/checkpoint-10125"),
            )
        ),
        "config_path": Path(
            os.getenv(
                "ROBUST_LSTM_CONFIG_PATH",
                str(SRC_DIR / "configs/lstm.yaml"),
            )
        ),
    },
    "cnn_a2": {},
}

TOP_K_PER_CLASS = 6
RANDOM_SEED = 67


@dataclass
class A2CNNRuntimeConfig:
    """Runtime settings for the A2 CNN adapter."""

    max_seq_length: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    embed_dim: int = 512
    num_filters: int = 300
    kernel_sizes: tuple[int, ...] = (3, 5, 7)
    num_epochs: int = 15
    batch_size: int = 512
    early_stopping_patience: int = 2


class _FixedSeqDataset(Dataset):
    """Fixed-length token-id dataset for CNN training."""

    def __init__(self, input_ids: list[list[int]], labels: list[int]):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class A2CNNAdapter(nn.Module):
    """Adapter that gives A2 CNN a shared inference API."""

    def __init__(
        self,
        cnn_model: nn.Module,
        tokenizer,
        max_seq_length: int,
        eval_batch_size: int,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.per_device_eval_batch_size = eval_batch_size

    def tokenize(self, sample: dict) -> dict:
        ids = self.tokenizer(sample["text"])
        ids = ids[: self.max_seq_length]
        if len(ids) < self.max_seq_length:
            ids = ids + [0] * (self.max_seq_length - len(ids))
        return {"input_ids": ids}

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: D401
        logits = self.cnn_model(input_ids, None)
        return SimpleNamespace(logits=logits)


def _load_module_from_path(module_name: str, module_path: Path):
    """Import a module from an absolute path."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {module_name} at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_a2_runtime_modules():
    """Load assignment2 tokenizer/CNN modules."""
    tok_module = _load_module_from_path(
        "assignment2_tokanizer_runtime", A2_SRC_DIR / "utils" / "tokanizer.py"
    )
    cnn_module = _load_module_from_path(
        "assignment2_cnn_runtime", A2_SRC_DIR / "models" / "cnn.py"
    )
    return tok_module, cnn_module


def _encode_fixed_len(texts: list[str], tokenizer, max_seq_length: int) -> list[list[int]]:
    """Tokenize and pad/truncate to a fixed sequence length."""
    encoded: list[list[int]] = []
    for text in tqdm(texts, desc="Encoding A2 CNN texts", leave=False):
        ids = tokenizer(text)
        ids = ids[:max_seq_length]
        if len(ids) < max_seq_length:
            ids = ids + [0] * (max_seq_length - len(ids))
        encoded.append(ids)
    return encoded


def _evaluate_a2_cnn(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Evaluate A2 CNN and return loss/metrics."""
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            logits = model(x, None)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            total_n += y.size(0)
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
    }


def _train_or_load_a2_cnn(train_raw, dev_raw):
    """Load cached A2 CNN model or train it."""
    cfg = A2CNNRuntimeConfig()
    tok_module, cnn_module = _load_a2_runtime_modules()
    BPETokenizer = tok_module.BPETokenizer
    CNN = cnn_module.CNN

    if A2_CNN_MODEL_PATH.exists() and A2_CNN_TOKENIZER_PATH.exists() and A2_CNN_META_PATH.exists():
        logger.info("Using cached A2 CNN artifacts: %s", A2_CNN_ARTIFACT_DIR)
        with A2_CNN_TOKENIZER_PATH.open("rb") as f:
            tokenizer = pickle.load(f)
        meta = json.loads(A2_CNN_META_PATH.read_text(encoding="utf-8"))
        runtime_cfg = A2CNNRuntimeConfig(
            max_seq_length=int(meta["max_seq_length"]),
            learning_rate=float(meta["learning_rate"]),
            weight_decay=float(meta["weight_decay"]),
            embed_dim=int(meta["embed_dim"]),
            num_filters=int(meta["num_filters"]),
            kernel_sizes=tuple(int(k) for k in meta["kernel_sizes"]),
            num_epochs=int(meta["num_epochs"]),
            batch_size=int(meta["batch_size"]),
            early_stopping_patience=int(meta.get("early_stopping_patience", 2)),
        )
        model = CNN(
            vocab_size=int(meta["vocab_size"]),
            embed_dim=runtime_cfg.embed_dim,
            num_filters=runtime_cfg.num_filters,
            kernel_sizes=list(runtime_cfg.kernel_sizes),
            num_classes=4,
        )
        model.load_state_dict(torch.load(A2_CNN_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        adapter = A2CNNAdapter(
            cnn_model=model,
            tokenizer=tokenizer,
            max_seq_length=runtime_cfg.max_seq_length,
            eval_batch_size=runtime_cfg.batch_size,
        ).to(DEVICE)
        adapter.eval()
        return adapter, SimpleNamespace(per_device_eval_batch_size=runtime_cfg.batch_size)

    logger.info("Training A2-style CNN for slice comparison...")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    tokenizer = BPETokenizer(min_freq=2, vocab_size=5000)
    tokenizer._build_vocab(train_raw["text"])
    vocab_size = max(int(v) for v in tokenizer.vocab.values()) + 1

    train_ids = _encode_fixed_len(train_raw["text"], tokenizer, cfg.max_seq_length)
    dev_ids = _encode_fixed_len(dev_raw["text"], tokenizer, cfg.max_seq_length)
    train_labels = [int(y) for y in train_raw["labels"]]
    dev_labels = [int(y) for y in dev_raw["labels"]]

    train_ds = _FixedSeqDataset(train_ids, train_labels)
    dev_ds = _FixedSeqDataset(dev_ids, dev_labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    model = CNN(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_filters=cfg.num_filters,
        kernel_sizes=list(cfg.kernel_sizes),
        num_classes=4,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_dev_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_seen = 0
        for batch in tqdm(train_loader, desc=f"A2 CNN train epoch {epoch}", leave=False):
            x = batch["input_ids"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(x, None)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * y.size(0)
            total_seen += y.size(0)

        train_loss = running_loss / max(total_seen, 1)
        dev_metrics = _evaluate_a2_cnn(model, dev_loader)
        logger.info(
            "A2 CNN epoch %d/%d | train_loss=%.4f dev_loss=%.4f dev_acc=%.4f dev_macro_f1=%.4f",
            epoch,
            cfg.num_epochs,
            train_loss,
            dev_metrics["loss"],
            dev_metrics["accuracy"],
            dev_metrics["macro_f1"],
        )

        if dev_metrics["loss"] < best_dev_loss - 1e-4:
            best_dev_loss = dev_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                logger.info("A2 CNN early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    A2_CNN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), A2_CNN_MODEL_PATH)
    with A2_CNN_TOKENIZER_PATH.open("wb") as f:
        pickle.dump(tokenizer, f)
    A2_CNN_META_PATH.write_text(
        json.dumps(
            {
                "max_seq_length": cfg.max_seq_length,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "embed_dim": cfg.embed_dim,
                "num_filters": cfg.num_filters,
                "kernel_sizes": list(cfg.kernel_sizes),
                "num_epochs": cfg.num_epochs,
                "batch_size": cfg.batch_size,
                "early_stopping_patience": cfg.early_stopping_patience,
                "vocab_size": vocab_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved A2 CNN artifacts: %s", A2_CNN_ARTIFACT_DIR)

    adapter = A2CNNAdapter(
        cnn_model=model,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        eval_batch_size=cfg.batch_size,
    ).to(DEVICE)
    adapter.eval()
    return adapter, SimpleNamespace(per_device_eval_batch_size=cfg.batch_size)


def _predict(model, dataset, batch_size: int) -> np.ndarray:
    """Run batched inference and return predicted class ids."""
    preds: list[int] = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
        batch = dataset[i : i + batch_size]
        input_tensors = [
            torch.tensor(ids, dtype=torch.long) for ids in batch["input_ids"]
        ]
        if "attention_mask" in batch:
            mask_tensors = [
                torch.tensor(mask, dtype=torch.long) for mask in batch["attention_mask"]
            ]
        else:
            mask_tensors = [
                torch.tensor([1 if token_id != 0 else 0 for token_id in ids])
                for ids in batch["input_ids"]
            ]
        attention_mask = pad_sequence(
            mask_tensors, batch_first=True, padding_value=0
        ).to(DEVICE)
        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=0).to(
            DEVICE
        )
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.int64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy, macro-F1, and weighted-F1."""
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _build_mask_pattern(words: list[str]) -> re.Pattern:
    """Build case-insensitive exact-word regex for masking."""
    escaped = sorted({w.lower() for w in words if w}, key=len, reverse=True)
    if not escaped:
        return re.compile(r"$^")
    return re.compile(r"\b(" + "|".join(map(re.escape, escaped)) + r")\b", re.IGNORECASE)


def _mask_text(text: str, pattern: re.Pattern, mask_token: str = "[MASK]") -> str:
    """Apply regex masking to text."""
    return pattern.sub(mask_token, text)


def _count_mask_hits(texts: list[str], pattern: re.Pattern) -> dict[str, int]:
    """Count how often a masking pattern appears in a text collection."""
    doc_hits = 0
    total_hits = 0
    for text in texts:
        hits = len(pattern.findall(text))
        total_hits += hits
        if hits > 0:
            doc_hits += 1
    return {"documents_touched": doc_hits, "total_token_matches": total_hits}


def _compute_length_buckets(train_texts: list[str]) -> tuple[list[float], np.ndarray]:
    """Compute quartile edges from train texts and return train counts."""
    train_counts = np.asarray([len(t.split()) for t in train_texts], dtype=np.int64)
    q1, q2, q3 = np.quantile(train_counts, [0.25, 0.5, 0.75])
    return [float(q1), float(q2), float(q3)], train_counts


def _assign_length_bucket(word_count: int, edges: list[float]) -> str:
    """Assign sample to Q1..Q4 using precomputed edges."""
    q1, q2, q3 = edges
    if word_count <= q1:
        return "Q1_shortest"
    if word_count <= q2:
        return "Q2_short"
    if word_count <= q3:
        return "Q3_long"
    return "Q4_longest"


def _derive_keywords_from_train(
    train_texts: list[str], train_labels: list[int], label_names: list[str]
) -> dict[str, list[str]]:
    """Derive class-indicative terms from train set only via TF-IDF delta scores."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ngram_range=(1, 1),
    )
    X = vectorizer.fit_transform(train_texts)
    terms = vectorizer.get_feature_names_out()
    y = np.asarray(train_labels, dtype=np.int64)

    keywords_by_class: dict[str, list[str]] = {}
    for class_id, class_name in enumerate(label_names):
        class_mask = y == class_id
        other_mask = y != class_id
        class_mean = np.asarray(X[class_mask].mean(axis=0)).ravel()
        other_mean = np.asarray(X[other_mask].mean(axis=0)).ravel()
        deltas = class_mean - other_mean

        idx_sorted = np.argsort(deltas)[::-1]
        picks: list[str] = []
        for idx in idx_sorted:
            if deltas[idx] <= 0:
                break
            token = terms[idx]
            if len(token) < 3:
                continue
            picks.append(token)
            if len(picks) >= TOP_K_PER_CLASS:
                break
        keywords_by_class[class_name] = picks
    return keywords_by_class


def _build_frequency_matched_control_terms(
    train_texts: list[str], keyword_terms: list[str], rng: np.random.Generator
) -> list[str]:
    """Sample random control terms matched by train-set document frequency."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ngram_range=(1, 1),
    )
    X = vectorizer.fit_transform(train_texts)
    terms = vectorizer.get_feature_names_out()
    term_to_df = {
        term: int(df)
        for term, df in zip(terms, np.asarray((X > 0).sum(axis=0)).ravel(), strict=True)
    }

    keyword_df = [term_to_df[t] for t in keyword_terms if t in term_to_df]
    if not keyword_df:
        raise RuntimeError("Could not compute train document-frequency for keyword terms.")
    min_df = min(keyword_df)
    max_df = max(keyword_df)

    control_pool = [
        t
        for t, df in term_to_df.items()
        if t not in set(keyword_terms) and min_df <= df <= max_df
    ]
    if len(control_pool) < len(keyword_terms):
        # Relax matching constraints if the pool is too small.
        median_df = int(np.median(keyword_df))
        control_pool = [
            t
            for t, df in term_to_df.items()
            if t not in set(keyword_terms) and abs(df - median_df) <= median_df
        ]
    if len(control_pool) < len(keyword_terms):
        raise RuntimeError("Not enough control terms after frequency matching.")

    return sorted(
        rng.choice(control_pool, size=len(keyword_terms), replace=False).tolist()
    )


def run(
    output_dir: Path = OUT_DIR,
    max_samples: int | None = None,
    eval_max_samples: int | None = None,
    models: tuple[str, ...] = ("lstm", "bert", "cnn_a2"),
) -> None:
    """Run slices and save JSON/Markdown outputs."""
    rng = np.random.default_rng(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = OmegaConf.load(SRC_DIR / "configs/dataset.yaml").dataset
    dataset_cfg.max_samples = max_samples
    dataset_cfg.eval_max_samples = eval_max_samples
    data = dataset_prep(dataset_cfg)

    train_raw = data["train"]
    dev_raw = data["dev"]
    test_raw = data["test"]
    label_names = [dataset_cfg.label_mapping[k] for k in sorted(dataset_cfg.label_mapping)]

    # Slice A: length buckets based on train quartiles
    length_edges, _ = _compute_length_buckets(train_raw["text"])
    test_counts = np.asarray([len(t.split()) for t in test_raw["text"]], dtype=np.int64)
    test_bucket_names = np.asarray(
        [_assign_length_bucket(int(c), length_edges) for c in test_counts]
    )

    # Slice B: keyword masking derived from train-only signal + random control
    keywords_by_class = _derive_keywords_from_train(
        train_texts=train_raw["text"],
        train_labels=train_raw["labels"],
        label_names=label_names,
    )
    keyword_terms = sorted({w for ws in keywords_by_class.values() for w in ws})
    random_terms = _build_frequency_matched_control_terms(
        train_texts=train_raw["text"], keyword_terms=keyword_terms, rng=rng
    )

    keyword_pattern = _build_mask_pattern(keyword_terms)
    random_pattern = _build_mask_pattern(random_terms)
    keyword_mask_hits = _count_mask_hits(test_raw["text"], keyword_pattern)
    random_mask_hits = _count_mask_hits(test_raw["text"], random_pattern)

    y_true = np.asarray(test_raw["labels"], dtype=np.int64)
    length_results: dict[str, dict] = {}
    masking_results: dict[str, dict] = {
        "method": {
            "top_k_per_class": TOP_K_PER_CLASS,
            "random_seed": RANDOM_SEED,
            "keywords_by_class": keywords_by_class,
            "keyword_terms": keyword_terms,
            "random_control_terms": random_terms,
            "mask_coverage_on_test": {
                "keyword_terms": keyword_mask_hits,
                "random_control_terms": random_mask_hits,
            },
        }
    }

    for model_name in models:
        logger.info("Running slices for %s...", model_name)
        if model_name == "cnn_a2":
            model, model_cfg = _train_or_load_a2_cnn(train_raw=train_raw, dev_raw=dev_raw)
        else:
            spec = MODEL_SPECS[model_name]
            model, model_cfg = load_model(
                model_type=model_name,
                model_path=str(spec["model_path"]),
                config_path=str(spec["config_path"]),
            )
        batch_size = int(getattr(model_cfg, "per_device_eval_batch_size", 16))

        test_regular = test_raw.map(
            model.tokenize, batched=False, load_from_cache_file=False
        )
        y_pred_regular = _predict(model, test_regular, batch_size=batch_size)
        regular_metrics = _metrics(y_true, y_pred_regular)

        # Length buckets on regular predictions
        bucket_table: dict[str, dict] = {}
        for bucket in ("Q1_shortest", "Q2_short", "Q3_long", "Q4_longest"):
            idx = np.where(test_bucket_names == bucket)[0]
            bucket_table[bucket] = {"count": int(len(idx)), **_metrics(y_true[idx], y_pred_regular[idx])}
        length_results[model_name] = {
            "length_edges_words": length_edges,
            "overall_regular": regular_metrics,
            "bucket_metrics": bucket_table,
        }

        # Keyword masking set
        test_keyword_masked = test_raw.map(
            lambda sample: {"text": _mask_text(sample["text"], keyword_pattern)},
            batched=False,
            load_from_cache_file=False,
        )
        test_keyword_masked = test_keyword_masked.map(
            model.tokenize, batched=False, load_from_cache_file=False
        )
        y_pred_keyword = _predict(model, test_keyword_masked, batch_size=batch_size)
        keyword_metrics = _metrics(y_true, y_pred_keyword)

        # Random control masking set
        test_random_masked = test_raw.map(
            lambda sample: {"text": _mask_text(sample["text"], random_pattern)},
            batched=False,
            load_from_cache_file=False,
        )
        test_random_masked = test_random_masked.map(
            model.tokenize, batched=False, load_from_cache_file=False
        )
        y_pred_random = _predict(model, test_random_masked, batch_size=batch_size)
        random_metrics = _metrics(y_true, y_pred_random)

        masking_results[model_name] = {
            "regular": regular_metrics,
            "keyword_masked": keyword_metrics,
            "random_masked_control": random_metrics,
            "drop_keyword": {
                k: regular_metrics[k] - keyword_metrics[k] for k in regular_metrics
            },
            "drop_random_control": {
                k: regular_metrics[k] - random_metrics[k] for k in regular_metrics
            },
            "excess_drop_keyword_minus_random": {
                k: (regular_metrics[k] - keyword_metrics[k]) - (regular_metrics[k] - random_metrics[k])
                for k in regular_metrics
            },
        }

    (output_dir / "length_buckets.json").write_text(
        json.dumps(length_results, indent=2), encoding="utf-8"
    )
    (output_dir / "keyword_masking.json").write_text(
        json.dumps(masking_results, indent=2), encoding="utf-8"
    )

    lines = [
        "# Robustness Summary (autogenerated)",
        "",
        "Setup:",
        f"- Length buckets use train quartiles (word-count edges): {length_edges}",
        f"- Keywords use train only TF-IDF deltas (top {TOP_K_PER_CLASS} per class)",
        "- Random control words are frequency matched on train data",
        f"- Masked keywords (n={len(keyword_terms)}): {', '.join(keyword_terms)}",
        f"- Random control words (n={len(random_terms)}): {', '.join(random_terms)}",
        (
            "- Mask coverage on test: keyword docs="
            f"{keyword_mask_hits['documents_touched']}, keyword matches={keyword_mask_hits['total_token_matches']}, "
            f"random docs={random_mask_hits['documents_touched']}, random matches={random_mask_hits['total_token_matches']}"
        ),
        "",
        "## Slice 1: Length buckets",
        "",
        "| Model | Bucket | Count | Accuracy | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model_name in models:
        for bucket in ("Q1_shortest", "Q2_short", "Q3_long", "Q4_longest"):
            row = length_results[model_name]["bucket_metrics"][bucket]
            lines.append(
                f"| {model_name} | {bucket} | {row['count']} | "
                f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
            )

    lines += [
        "",
        "## Slice 2: Keyword masking with random control",
        "",
        "| Model | Regular Macro-F1 | Keyword-Masked Macro-F1 | Random-Control Macro-F1 | Keyword Drop | Random Drop | Excess Drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in models:
        row = masking_results[model_name]
        lines.append(
            f"| {model_name} | {row['regular']['macro_f1']:.4f} | "
            f"{row['keyword_masked']['macro_f1']:.4f} | "
            f"{row['random_masked_control']['macro_f1']:.4f} | "
            f"{row['drop_keyword']['macro_f1']:.4f} | "
            f"{row['drop_random_control']['macro_f1']:.4f} | "
            f"{row['excess_drop_keyword_minus_random']['macro_f1']:.4f} |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved slice outputs to %s", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run robustness slices.")
    parser.add_argument(
        "--output-dir",
        default="robustness_slices",
        help="Output directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional train cap.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Optional dev/test cap.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bert", "lstm", "cnn_a2"],
        default=["lstm", "bert", "cnn_a2"],
        help="Models to include.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = SRC_DIR / out_dir
    run(
        output_dir=out_dir,
        max_samples=args.max_samples,
        eval_max_samples=args.eval_max_samples,
        models=tuple(args.models),
    )
