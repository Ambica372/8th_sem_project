"""
=============================================================================
logger.py  —  Structured Experiment Logging Utility
=============================================================================

Features
--------
- Tee all print/log output to both terminal AND a .log file simultaneously.
- Record per-epoch training metrics (loss, accuracy, val_loss, val_accuracy).
- Persist epoch logs to CSV and JSON inside a /logs sub-directory.
- Generate a Markdown (.md) summary report on demand.
- Safe append mode: existing logs are never overwritten unless requested.
- Works identically on Google Colab and local terminal.

Usage
-----
    from logger import ExperimentLogger

    logger = ExperimentLogger(
        output_dir="obj2",          # top-level output folder
        experiment_name="dl_run",   # used as file prefix
        config=CONFIG,              # dict of hyperparameters / paths
        overwrite=False,            # True to start fresh
    )

    # Tee all prints
    logger.start_tee()

    # During / after Keras model.fit():
    logger.record_epoch(fold=1, epoch=3, history_dict=hist.history, epoch_idx=2)

    # At end of training:
    logger.save_epoch_logs()         # writes CSV + JSON
    logger.generate_md_report(fold_df, summary_df)

    # Stop capturing stdout
    logger.stop_tee()
=============================================================================
"""

from __future__ import annotations

import csv
import json
import os
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Tee  —  duplicate stdout to a file while keeping terminal output
# ---------------------------------------------------------------------------

class _Tee(io.TextIOBase):
    """Wraps sys.stdout so every write goes to both terminal and a log file."""

    def __init__(self, original_stdout, log_file_handle):
        self._stdout  = original_stdout
        self._logfile = log_file_handle

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._logfile.write(data)
        self._stdout.flush()
        self._logfile.flush()
        return len(data)

    def flush(self):
        self._stdout.flush()
        self._logfile.flush()

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Structured experiment logger.

    Parameters
    ----------
    output_dir      : Root output folder (e.g. "obj2").
    experiment_name : Short tag used as filename prefix (e.g. "dl_run").
    config          : Dict of training configuration / hyper-parameters.
    overwrite       : If False (default), existing logs are appended.
                      If True, all log files in this run are recreated.
    """

    def __init__(
        self,
        output_dir: str | Path,
        experiment_name: str,
        config: Dict[str, Any],
        overwrite: bool = False,
    ):
        self.experiment_name = experiment_name
        self.config          = config
        self.overwrite       = overwrite

        # Create logs sub-directory
        self.log_dir = Path(output_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped run id  (used in filenames when not overwriting)
        self._run_id = config.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

        # File paths --------------------------------------------------------
        suffix = "" if overwrite else f"_{self._run_id}"
        self._log_txt   = self.log_dir / f"{experiment_name}{suffix}.log"
        self._epoch_csv = self.log_dir / f"{experiment_name}{suffix}_epochs.csv"
        self._epoch_json= self.log_dir / f"{experiment_name}{suffix}_epochs.json"
        self._md_report = self.log_dir / f"{experiment_name}{suffix}_report.md"

        # In-memory stores --------------------------------------------------
        self._console_records: List[Dict] = []   # {timestamp, level, message}
        self._epoch_records:   List[Dict] = []   # per-epoch metric rows

        # Tee handles (managed via start_tee / stop_tee)
        self._tee_obj:    Optional[_Tee] = None
        self._log_handle = None
        self._orig_stdout = None

        # Initialise / open the console log file
        mode = "w" if overwrite else "a"
        self._log_handle = open(self._log_txt, mode, encoding="utf-8", buffering=1)
        self._log_handle.write(
            f"\n{'='*70}\n"
            f"Run started : {datetime.now().isoformat()}\n"
            f"Experiment  : {experiment_name}\n"
            f"{'='*70}\n"
        )

        # Initialise epoch CSV (write header only if file is new / overwrite)
        self._epoch_csv_header_written = self._epoch_csv.exists() and not overwrite

    # ------------------------------------------------------------------
    # Tee control
    # ------------------------------------------------------------------

    def start_tee(self):
        """Redirect sys.stdout so all print() calls also go to the log file."""
        if self._tee_obj is not None:
            return  # already started
        self._orig_stdout = sys.stdout
        self._tee_obj     = _Tee(self._orig_stdout, self._log_handle)
        sys.stdout        = self._tee_obj

    def stop_tee(self):
        """Restore original sys.stdout and flush the log file."""
        if self._orig_stdout is not None:
            sys.stdout        = self._orig_stdout
            self._orig_stdout = None
        self._tee_obj = None
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.flush()

    # ------------------------------------------------------------------
    # Console log helpers
    # ------------------------------------------------------------------

    def log(self, msg: str, level: str = "INFO"):
        """Print a timestamped message and append to the in-memory log."""
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{ts}] [{level}] {msg}"
        self._console_records.append({"timestamp": ts, "level": level, "message": msg})
        # print() goes through tee (if active) or direct stdout
        print(formatted)

    def save_console_log(self):
        """Write console records to CSV (append-safe)."""
        csv_path = self.log_dir / f"{self.experiment_name}_console.csv"
        mode     = "w" if self.overwrite else "a"
        write_header = self.overwrite or not csv_path.exists()
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "level", "message"])
            if write_header:
                writer.writeheader()
            writer.writerows(self._console_records)

    # ------------------------------------------------------------------
    # Epoch metric recording
    # ------------------------------------------------------------------

    def record_epoch(
        self,
        fold: int,
        epoch: int,            # 1-indexed epoch number
        history_dict: Dict,    # Keras hist.history (full list) or single-epoch dict
        epoch_idx: int = -1,   # which index in the lists to read (-1 = last)
    ):
        """
        Append one epoch's metrics to the in-memory epoch log.

        Pass `history_dict` as `hist.history` after model.fit() completes,
        and epoch_idx as the epoch you want to record (0-based inside the list).
        Use epoch_idx=-1 to record the final epoch automatically.
        """
        def _get(key: str):
            lst = history_dict.get(key, [])
            if not lst:
                return None
            idx = epoch_idx if epoch_idx >= 0 else len(lst) - 1
            idx = min(idx, len(lst) - 1)
            return float(lst[idx])

        row = {
            "run_id"        : self._run_id,
            "fold"          : fold,
            "epoch"         : epoch,
            "timestamp"     : datetime.now().isoformat(),
            "loss"          : _get("loss"),
            "accuracy"      : _get("accuracy"),
            "val_loss"      : _get("val_loss"),
            "val_accuracy"  : _get("val_accuracy"),
            "lr"            : _get("lr"),
        }
        self._epoch_records.append(row)

    def record_all_epochs(self, fold: int, history_dict: Dict):
        """
        Convenience: record every epoch in a completed Keras history dict.

        Call once per fold after model.fit() returns.
        """
        n_epochs = len(history_dict.get("loss", []))
        for i in range(n_epochs):
            self.record_epoch(fold=fold, epoch=i + 1,
                              history_dict=history_dict, epoch_idx=i)

    def save_epoch_logs(self):
        """Persist epoch records to both CSV and JSON."""
        if not self._epoch_records:
            self.log("No epoch records to save.", "WARN")
            return

        df = pd.DataFrame(self._epoch_records)

        # — CSV (append-safe) —
        mode         = "w" if (self.overwrite or not self._epoch_csv.exists()) else "a"
        write_header = mode == "w" or not self._epoch_csv.exists()
        df.to_csv(self._epoch_csv, mode=mode, header=write_header, index=False)
        self.log(f"Epoch CSV → {self._epoch_csv}")

        # — JSON (always write full snapshot for this run) —
        existing = []
        if self._epoch_json.exists() and not self.overwrite:
            try:
                with open(self._epoch_json, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []
        combined = existing + self._epoch_records
        with open(self._epoch_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        self.log(f"Epoch JSON → {self._epoch_json}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def generate_md_report(
        self,
        fold_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        extra_notes: str = "",
    ) -> Path:
        """
        Write a Markdown report summarising the full training run.

        Parameters
        ----------
        fold_df     : DataFrame with per-fold accuracy / F1.
        summary_df  : DataFrame with mean ± std aggregates.
        extra_notes : Optional free-text appended at the end.

        Returns
        -------
        Path to the generated .md file.
        """
        s = summary_df.iloc[0]
        lines: List[str] = []

        # Header
        lines += [
            f"# Experiment Report: `{self.experiment_name}`",
            "",
            f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"> **Run ID:** `{self._run_id}`",
            "",
            "---",
            "",
        ]

        # Training Configuration
        lines += [
            "## 1. Training Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
        ]
        for k, v in self.config.items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

        # Model description (static for this pipeline)
        lines += [
            "## 2. Model Architecture",
            "",
            "| Branch | Description |",
            "|--------|-------------|",
            "| **EEG** | Input(time, 62 ch) → Conv1D(64,k7) → BN → Pool → Conv1D(128,k5) → BN → Pool → Conv1D(256,k3) → GlobalAvgPool → Dense(128) → Dropout(0.4) → Dense(64) |",
            "| **Eye** | Input(5 feats) → Dense(64) → BN → Dropout(0.3) → Dense(32) |",
            "| **Fusion** | Concatenate → Dense(128) → Dropout(0.4) → Dense(64) → Softmax(4) |",
            "",
        ]

        # Final / Summary metrics
        lines += [
            "## 3. Final Metrics (Cross-Validated)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Accuracy | **{s['mean_acc']:.4f}** |",
            f"| Std Accuracy  | {s['std_acc']:.4f} |",
            f"| Mean F1 (weighted) | **{s['mean_f1']:.4f}** |",
            f"| Std F1 (weighted)  | {s['std_f1']:.4f} |",
            f"| Folds | {int(s['n_folds'])} |",
            "",
        ]

        # Per-epoch summary per fold (last epoch values)
        if self._epoch_records:
            epoch_df = pd.DataFrame(self._epoch_records)
            lines += [
                "## 4. Per-Epoch Metrics (All Folds)",
                "",
                "| Fold | Epoch | Loss | Accuracy | Val Loss | Val Accuracy |",
                "|------|-------|------|----------|----------|--------------|",
            ]
            for _, row in epoch_df.iterrows():
                loss     = f"{row['loss']:.4f}"     if row['loss']     is not None else "—"
                acc      = f"{row['accuracy']:.4f}" if row['accuracy'] is not None else "—"
                val_loss = f"{row['val_loss']:.4f}" if row['val_loss'] is not None else "—"
                val_acc  = f"{row['val_accuracy']:.4f}" if row['val_accuracy'] is not None else "—"
                lines.append(
                    f"| {int(row['fold'])} | {int(row['epoch'])} "
                    f"| {loss} | {acc} | {val_loss} | {val_acc} |"
                )
            lines.append("")

        # Per-fold summary table
        lines += [
            "## 5. Per-Fold Summary",
            "",
            "| Fold | Accuracy | F1 (Weighted) | Train Size | Test Size | Test Subjects |",
            "|------|----------|--------------|------------|-----------|---------------|",
        ]
        for _, r in fold_df.iterrows():
            lines.append(
                f"| {int(r['fold'])} | {r['accuracy']:.4f} | {r['f1_weighted']:.4f} "
                f"| {int(r['train_size'])} | {int(r['test_size'])} | {r['test_subjects']} |"
            )
        lines.append("")

        # Output file listing
        lines += [
            "## 6. Output Files",
            "",
            f"| File | Description |",
            f"|------|-------------|",
            f"| `logs/{self._epoch_csv.name}` | Per-epoch metrics (CSV) |",
            f"| `logs/{self._epoch_json.name}` | Per-epoch metrics (JSON) |",
            f"| `logs/{self._log_txt.name}` | Full console log (tee) |",
            f"| `fold_metrics_dl.csv` | Per-fold accuracy + F1 |",
            f"| `summary_metrics_dl.csv` | Mean ± std summary |",
            f"| `config_dl.json` | Saved configuration |",
            f"| `dl_report.pdf` | Full PDF report |",
            "",
        ]

        if extra_notes:
            lines += ["## 7. Notes", "", extra_notes, ""]

        lines += ["---", f"*Report auto-generated by `logger.py`*", ""]

        report_text = "\n".join(lines)
        with open(self._md_report, "w", encoding="utf-8") as f:
            f.write(report_text)

        self.log(f"Markdown report → {self._md_report}")
        return self._md_report

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Flush and close all open file handles."""
        self.stop_tee()
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.write(
                f"\nRun ended: {datetime.now().isoformat()}\n")
            self._log_handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
