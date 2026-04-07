from pathlib import Path
import subprocess

from bitescore.api import main as api_main
from bitescore.tools import localcolabfold
from bitescore.tools.localcolabfold import localcolabfold_status


def test_localcolabfold_status_binary_override(tmp_path, monkeypatch):
    fake_bin = tmp_path / "localcolabfold"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.delenv("LOCALCOLABFOLD_BIN", raising=False)

    status = localcolabfold_status(binary_override=str(fake_bin))

    assert status["available"] is True
    assert status["binary"] == str(fake_bin)
    assert status["resolved_path"] == str(fake_bin)


def test_run_pipeline_sync_passes_localcolabfold_options(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_load_config(_yaml, overrides):
        captured.update(overrides)
        return {"outdir": str(tmp_path), "input_type": "proteome"}

    monkeypatch.setattr(api_main, "load_config", _fake_load_config)
    monkeypatch.setattr(api_main, "run_pipeline", lambda cfg: None)
    monkeypatch.setattr(api_main, "_collect_pipeline_outputs", lambda outdir, input_type: {"ok": True})

    result = api_main._run_pipeline_sync(
        input_path=Path("input.faa"),
        input_type="proteome",
        organism=None,
        outdir=tmp_path / "results",
        opts={
            "localcolabfold_timeout": "42",
            "localcolabfold_bin": "  /opt/localcolabfold  ",
        },
    )

    assert result == {"ok": True}
    assert captured["localcolabfold_timeout"] == 42
    assert captured["localcolabfold_bin"] == "/opt/localcolabfold"


def test_predict_structure_adds_stable_flags_for_colabfold_batch(monkeypatch, tmp_path):
    observed_cmd: dict[str, list[str]] = {}
    fake_bin = tmp_path / "colabfold_batch"
    fake_bin.write_text("#!/bin/sh\n")

    monkeypatch.setattr(localcolabfold.shutil, "which", lambda _: str(fake_bin))

    def _fake_run(cmd, **kwargs):
        observed_cmd["cmd"] = cmd
        output_dir = Path(cmd[2])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ranked_0.pdb").write_text("MODEL\nEND\n")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(localcolabfold.subprocess, "run", _fake_run)

    result = localcolabfold.predict_structure(
        sequence="MSTNPKPQRITK",
        seq_id="seq1",
        cache_dir=tmp_path / "cache",
        binary=str(fake_bin),
    )

    assert result is not None
    cmd = observed_cmd["cmd"]
    assert "--num-recycle" in cmd
    assert "--num-models" in cmd
    assert "--disable-unified-memory" in cmd

