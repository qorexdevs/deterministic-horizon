from __future__ import annotations

import builtins
import sys
import types

import pytest
from deterministic_horizon.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_train_reports_missing_training_extras(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("deterministic_horizon.training"):
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert "Training extras" in result.stdout


@pytest.fixture
def stub_training_stack(monkeypatch):
    """Import the finetune module without the heavy torch/transformers deps."""
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Dataset = object
    utils.data = data
    torch.utils = utils

    transformers = types.ModuleType("transformers")
    for name in (
        "AutoModelForCausalLM",
        "AutoTokenizer",
        "DataCollatorForLanguageModeling",
        "Trainer",
        "TrainingArguments",
    ):
        setattr(transformers, name, object)

    for name in ("torch", "torch.utils", "torch.utils.data", "transformers"):
        monkeypatch.setitem(
            sys.modules,
            name,
            {
                "torch": torch,
                "torch.utils": utils,
                "torch.utils.data": data,
                "transformers": transformers,
            }[name],
        )
    monkeypatch.delitem(sys.modules, "deterministic_horizon.training.finetune", raising=False)


def test_train_loads_config_and_calls_finetuning(tmp_path, monkeypatch, stub_training_stack):
    from deterministic_horizon.training import finetune

    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {"metrics": {"eval_loss": 0.42}}

    monkeypatch.setattr(finetune, "run_finetuning", fake_run)

    cfg_file = tmp_path / "finetune.yaml"
    cfg_file.write_text("model_name: test/tiny\nnum_epochs: 1\nbogus_key: 7\n")
    out_dir = tmp_path / "ckpt"

    result = runner.invoke(app, ["train", "--config", str(cfg_file), "--output-dir", str(out_dir)])

    assert result.exit_code == 0, result.stdout
    cfg = captured["config"]
    assert cfg.model_name == "test/tiny"
    assert cfg.num_epochs == 1
    assert cfg.output_dir == str(out_dir)
    assert "bogus_key" in result.stdout
