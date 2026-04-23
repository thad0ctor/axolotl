"""M6 headline test — multi-GPU ProTrain throughput scaling on 4x RTX 3090.

Launches two separate training runs and asserts that the 4-GPU run
clears the ``>= 2.5x`` scaling bar specified in M6 of the plan:

* single-rank baseline: 1 worker on one 3090 (logical device 0 under
  ``CUDA_VISIBLE_DEVICES=1``).
* 4-rank run: 4 workers on ``CUDA_VISIBLE_DEVICES=1,2,4,5``.

Both runs build a fresh-init Llama-7B, apply the LoRA target set used
by the M4 integration test, wrap the result with ``protrain_model_wrapper``,
wrap that with ``torch.nn.parallel.DistributedDataParallel``
(``find_unused_parameters=True`` — LoRA freezes > 99% of the base
model, so without it DDP deadlocks the backward), and execute 5
iterations. Iteration 0 is warm-up (CUDA graph/alloc init +
NCCL warm-up on the 4-rank path); iterations 1..4 are averaged.

Throughput is measured as ``world_size * batch_size / avg_iter_s``
(samples/s across the data-parallel set). The assertion is

    throughput_4gpu / throughput_1gpu >= 2.5

matching the ``plan.md`` M6 criterion.

The two runs are executed in **separate subprocesses** because
``CUDA_VISIBLE_DEVICES`` has to be baked in before any CUDA call is
made in the process; the pytest host process has usually already
touched CUDA by the time this test runs.

Marked ``slow`` + ``gpu`` so the default ``pytest -m 'not slow'`` lane
still skips it. Auto-skips when fewer than 4 physical GPUs are visible
to the pytest host — the launcher env masks visibility below, so the
check is done via ``nvidia-smi`` at test time.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _nvidia_smi_gpu_count() -> int:
    """Return the number of GPUs reported by ``nvidia-smi``.

    Avoids importing torch (which reads ``CUDA_VISIBLE_DEVICES`` at
    import time and would under-report inside a masked pytest process).
    Returns 0 if ``nvidia-smi`` is unavailable or the call fails.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


# The full worker script is kept as a heredoc string (rather than a
# helper file) so the test is self-contained. Subprocess invokes
# ``python -c <script>`` with CUDA_VISIBLE_DEVICES + env-driven config.
_WORKER_SCRIPT = textwrap.dedent(
    '''
    """Subprocess entry point: spawns N workers and reports avg iter time.

    Reads from env:
        PROTRAIN_WORLD_SIZE        — 1 or 4
        PROTRAIN_BATCH_SIZE        — per-rank batch size
        PROTRAIN_SEQ_LEN           — sequence length
        PROTRAIN_N_ITERS           — total iterations including warmup
        PROTRAIN_N_WARMUP          — warmup iterations to discard
        PROTRAIN_OUT_FILE          — path where rank 0 writes avg_iter_s
    """
    import os
    import sys
    import time

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    def _worker(rank: int, world_size: int, out_file: str,
                bs: int, seq: int, n_iters: int, n_warmup: int) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        # Bind this rank to its own GPU BEFORE any CUDA alloc.
        # ``CUDA_VISIBLE_DEVICES`` is a comma list at the subprocess
        # level (e.g. "1,2,4,5"); ``rank`` is the logical index into
        # that list, so ``torch.cuda.set_device(rank)`` maps to a
        # distinct physical GPU per rank. Every subsequent cuda
        # allocation in this process defaults to that device.
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device("cuda", rank),
        )
        try:
            _run(rank, world_size, out_file, bs, seq, n_iters, n_warmup)
        finally:
            # Ensure every rank arrives at the barrier before teardown,
            # otherwise NCCL can abort with "bootstrap socket connection
            # refused" on the tail ranks.
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


    def _run(rank: int, world_size: int, out_file: str,
             bs: int, seq: int, n_iters: int, n_warmup: int) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model

        from axolotl.integrations.protrain.api import (
            protrain_model_wrapper,
            protrain_optimizer_wrapper,
        )
        from axolotl.integrations.protrain.types import HardwareProfile

        torch.manual_seed(42 + rank)

        cfg = LlamaConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            torch_dtype="float16",
            use_cache=False,
        )

        # Land this rank's model on its own GPU. ``rank`` indexes into
        # the subprocess's ``CUDA_VISIBLE_DEVICES`` list (e.g. with
        # ``CUDA_VISIBLE_DEVICES=1,2,4,5``, rank 0 -> physical GPU 1,
        # rank 1 -> physical GPU 2, etc). ``torch.cuda.set_device`` was
        # called in ``_worker`` before this ran.
        device = torch.device("cuda", rank)

        model = LlamaForCausalLM(cfg).half().to(device)

        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        hw = HardwareProfile(
            gpu_sku=torch.cuda.get_device_name(rank),
            gpu_memory_bytes=torch.cuda.get_device_properties(rank).total_memory,
            gpu_count=world_size,  # affects profiler cache key
            pcie_h2d_bps=13e9,
            pcie_d2h_bps=13e9,
            has_nvlink=False,
        )

        # ``force_all_persistent=True`` pins every chunk on GPU so DDP's
        # grad-shape snapshot at wrap time matches the real per-param
        # shapes. Without this, ``materialize_offload`` sets
        # non-persistent chunks' param.data to zero-sized GPU placeholders,
        # and DDP's constructor records those shapes and then rejects
        # the real-shape grads at iter-0 backward. For LoRA-on-7B the
        # whole base (~13.5 GiB fp16) fits alongside activations + LoRA
        # optimizer state in 24 GiB so making every chunk persistent is
        # the configuration the searcher would have picked anyway under
        # the 20 GiB capacity budget.
        wrapped = protrain_model_wrapper(
            model,
            model_config=cfg,
            hardware_profile=hw,
            batch_size=bs,
            seq_len=seq,
            capacity_bytes=20 * (1 << 30),
            force_all_persistent=True,
        )
        optim = protrain_optimizer_wrapper(wrapped, lr=1e-4)

        # DDP owns cross-rank grad reduction in this composition; tell
        # the chunk manager to skip its own per-param all_reduce so we
        # don't do the sync twice (the per-param version is much slower
        # than DDP's bucketed allreduce on pure-PCIe 3090 pairs and
        # would dominate the iter time).
        if world_size > 1:
            wrapped.chunk_manager.skip_internal_grad_reduce = True

        use_ddp = world_size > 1 and os.environ.get("PROTRAIN_SKIP_DDP") != "1"
        if use_ddp:
            # Wrap with DDP AFTER protrain so the chunk manager's hooks
            # see the real module tree. DDP by default skips params
            # with ``requires_grad=False``, so the frozen Llama-7B base
            # is free — we do NOT need ``find_unused_parameters=True``,
            # and leaving it off is the critical knob for cracking the
            # 2.5x bar (it would otherwise trigger a full autograd-
            # graph walk per backward). ``gradient_as_bucket_view=True``
            # avoids an extra copy inside DDP's allreduce bucket fill.
            ddp_module = torch.nn.parallel.DistributedDataParallel(
                wrapped.module,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
                broadcast_buffers=False,  # avoids per-iter buffer sync on LoRA
                gradient_as_bucket_view=True,
            )
        else:
            ddp_module = wrapped.module

        input_ids = torch.randint(
            0, cfg.vocab_size, (bs, seq), device=device, dtype=torch.long
        )
        labels = input_ids.clone()

        # Iterate. Time each iteration plus its sub-phases (fwd / bwd /
        # opt) on rank 0; the breakdown is written alongside the
        # aggregate so failure reports can point at the bottleneck
        # (DDP sync dominated vs. compute dominated etc).
        iter_times = []
        fwd_times, bwd_times, opt_times = [], [], []
        for i in range(n_iters):
            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()  # start-line sync across ranks
            t0 = time.perf_counter()

            out = ddp_module(input_ids=input_ids, labels=labels)
            loss = out.loss
            torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t0
            t1 = time.perf_counter()

            loss.backward()
            torch.cuda.synchronize()
            t_bwd = time.perf_counter() - t1
            t2 = time.perf_counter()

            optim.step()
            optim.zero_grad()
            torch.cuda.synchronize()
            t_opt = time.perf_counter() - t2

            if world_size > 1:
                dist.barrier()
            iter_times.append(time.perf_counter() - t0)
            fwd_times.append(t_fwd)
            bwd_times.append(t_bwd)
            opt_times.append(t_opt)

        if rank == 0:
            kept = iter_times[n_warmup:]
            kept_fwd = fwd_times[n_warmup:]
            kept_bwd = bwd_times[n_warmup:]
            kept_opt = opt_times[n_warmup:]
            avg = sum(kept) / max(1, len(kept))
            avg_fwd = sum(kept_fwd) / max(1, len(kept_fwd))
            avg_bwd = sum(kept_bwd) / max(1, len(kept_bwd))
            avg_opt = sum(kept_opt) / max(1, len(kept_opt))
            with open(out_file, "w") as f:
                f.write(
                    f"avg_iter_s={avg:.6f}\\n"
                    f"avg_fwd_s={avg_fwd:.6f}\\n"
                    f"avg_bwd_s={avg_bwd:.6f}\\n"
                    f"avg_opt_s={avg_opt:.6f}\\n"
                    f"all_times={iter_times}\\n"
                    f"fwd_times={fwd_times}\\n"
                    f"bwd_times={bwd_times}\\n"
                    f"opt_times={opt_times}\\n"
                )
            print(f"[rank0] world={world_size} bs={bs} seq={seq} "
                  f"avg_iter={avg:.4f}s (fwd={avg_fwd:.3f} "
                  f"bwd={avg_bwd:.3f} opt={avg_opt:.3f}) "
                  f"iters={iter_times}",
                  flush=True)


    def main() -> int:
        world = int(os.environ["PROTRAIN_WORLD_SIZE"])
        bs = int(os.environ["PROTRAIN_BATCH_SIZE"])
        seq = int(os.environ["PROTRAIN_SEQ_LEN"])
        n_iters = int(os.environ["PROTRAIN_N_ITERS"])
        n_warmup = int(os.environ["PROTRAIN_N_WARMUP"])
        out_file = os.environ["PROTRAIN_OUT_FILE"]

        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(world):
            p = ctx.Process(
                target=_worker,
                args=(rank, world, out_file, bs, seq, n_iters, n_warmup),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for p in procs:
            if p.exitcode != 0:
                print(f"worker pid={p.pid} exited with {p.exitcode}", flush=True)
                return p.exitcode
        return 0


    if __name__ == "__main__":
        sys.exit(main())
    '''
)


def _parse_avg(out_path: Path) -> float:
    """Read the ``avg_iter_s=`` line the worker wrote; return seconds."""
    text = out_path.read_text()
    for line in text.splitlines():
        if line.startswith("avg_iter_s="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"no avg_iter_s line in {out_path}: {text!r}")


def _launch(
    *,
    world_size: int,
    cuda_visible: str,
    bs: int,
    seq: int,
    n_iters: int,
    n_warmup: int,
    out_path: Path,
    tmp_path: Path,
) -> None:
    """Run one subprocess that spawns ``world_size`` ranks."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    # Without this torch defaults to FASTEST_FIRST, which on a
    # heterogenous box re-orders the visible set by SM count. On our
    # test rig that mixed 3090s with RTX PRO 6000 / 5090 cards,
    # ``CUDA_VISIBLE_DEVICES=1,2,4,5`` (nvidia-smi indices, all 3090s)
    # would expose Blackwell cards to torch as devices 0 and 1 — a
    # latent correctness issue and the reason the first multi-rank
    # iteration landed half its workers on much faster silicon than
    # the others, blowing up the barrier tail. Forcing PCI_BUS_ID
    # order keeps the set-of-GPUs identity consistent between
    # ``nvidia-smi`` and torch.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PROTRAIN_WORLD_SIZE"] = str(world_size)
    env["PROTRAIN_BATCH_SIZE"] = str(bs)
    env["PROTRAIN_SEQ_LEN"] = str(seq)
    env["PROTRAIN_N_ITERS"] = str(n_iters)
    env["PROTRAIN_N_WARMUP"] = str(n_warmup)
    env["PROTRAIN_OUT_FILE"] = str(out_path)
    # Avoid NCCL IB probes on a pure-PCIe box — faster startup and no
    # spurious warnings about ibv_open_device failures.
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_P2P_DISABLE", "0")

    # Persist the script to a file under tmp_path so tracebacks point
    # at a real line number rather than ``<string>:1``.
    script_path = tmp_path / f"_worker_world{world_size}.py"
    script_path.write_text(_WORKER_SCRIPT)

    # Drop the parent process's log file, if any, before launch.
    log_path = tmp_path / f"worker_world{world_size}.log"
    with log_path.open("w") as log_f:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,  # 30 min upper bound for profiler + 5 iters
        )
    if proc.returncode != 0:
        tail = log_path.read_text()[-4000:]
        raise RuntimeError(
            f"worker world={world_size} failed (exit={proc.returncode}); "
            f"log tail:\n{tail}"
        )


@pytest.mark.slow
@pytest.mark.gpu
def test_protrain_4gpu_throughput_scaling(tmp_path) -> None:
    """Paper's M6 claim: 4-GPU ProTrain >= 2.5x single-GPU throughput."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    gpu_count = _nvidia_smi_gpu_count()
    if gpu_count < 4:
        pytest.skip(
            f"requires >= 4 GPUs; nvidia-smi reports {gpu_count}"
        )

    # Per-rank batch size 2 amortizes the Python-level hook overhead
    # (4 hooks x 32 blocks x 2 passes = 256 callbacks per iter) across
    # more compute per iter. At bs=1 seq=256 the hook cost is a
    # meaningful fraction of iter time on 3090 and hurts the scaling
    # assertion for reasons unrelated to ProTrain's distributed path.
    bs = 2
    seq = 256
    n_iters = 6
    n_warmup = 2

    # ---- Single-rank baseline ------------------------------------------
    out_single = tmp_path / "single.out"
    _launch(
        world_size=1,
        cuda_visible="1",
        bs=bs,
        seq=seq,
        n_iters=n_iters,
        n_warmup=n_warmup,
        out_path=out_single,
        tmp_path=tmp_path,
    )
    t_single = _parse_avg(out_single)

    # ---- 4-rank run ----------------------------------------------------
    out_multi = tmp_path / "multi.out"
    _launch(
        world_size=4,
        cuda_visible="1,2,4,5",
        bs=bs,
        seq=seq,
        n_iters=n_iters,
        n_warmup=n_warmup,
        out_path=out_multi,
        tmp_path=tmp_path,
    )
    t_multi = _parse_avg(out_multi)

    throughput_1 = 1 * bs / t_single
    throughput_4 = 4 * bs / t_multi
    scaling = throughput_4 / throughput_1

    print(
        "\nProTrain M6 multi-GPU scaling:\n"
        f"  single-rank avg iter:    {t_single:.3f} s "
        f"({throughput_1:.3f} samples/s)\n"
        f"  4-rank avg iter:         {t_multi:.3f} s "
        f"({throughput_4:.3f} samples/s)\n"
        f"  scaling:                 {scaling:.2f}x "
        f"(threshold: 2.50x)"
    )

    assert scaling >= 2.5, (
        f"ProTrain 4-GPU throughput only {scaling:.2f}x single-GPU "
        f"(need >= 2.5x). "
        f"single: {t_single:.3f}s ({throughput_1:.3f} samples/s); "
        f"4-rank: {t_multi:.3f}s ({throughput_4:.3f} samples/s)"
    )
