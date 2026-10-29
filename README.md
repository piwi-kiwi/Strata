# Strata Dual-GPU

A fork of [Strata](https://github.com/Niko1221/Strata) for a computer with two NVIDIA GPUs
and 32 GB of RAM. It runs the Qwen3.8-Flash-Next mixture-of-experts models locally, with both
cards working on the same request.

The fork is based on Strata 0.1.38. It was created before upstream added its own multi-GPU
support. Upstream now supports multiple cards, but its low-RAM mode is designed for one card;
this fork keeps the low-RAM mode working across a layer split and overlaps the two stages so
that neither card waits for the other.

## Branches

- `dualgpu` is the default branch: the dual-GPU fork described on this page.
- `main` tracks upstream Strata 0.2.7 without the dual-GPU changes.

The two branches are separate. Use `dualgpu` for the two-card setup. Use `main` only if you
need a feature that exists in upstream 0.2.7 and not in this fork.

## What this fork changes

- **Low-RAM mode on a layer split.** Each card keeps a cache of its own hottest experts; the
  rest are copied into system RAM. The model does not read experts from the SSD while it
  answers.
- **Pipelined windows.** The first card starts the next decoding step while the second card is
  still checking the current one. This is enabled by default when exactly two GPUs are used.
- **Faster card last.** The last stage also runs the output head and the draft layer, so the
  server orders the cards by speed when the split is automatic.
- **Per-card layer weights.** Each card loads only the dense weights of the layers it runs. The
  freed VRAM is used for that card's expert cache.
- **Asynchronous expert swaps.** Moving experts between VRAM and RAM no longer stops decoding
  while the copy runs.
- **Fewer host waits.** The stages synchronise through device-side flags, the draft chain runs
  in one CUDA graph, and RTX 50 cards use programmatic dependent launch.
- **Defaults for the host.** On hybrid Intel CPUs the expert pool uses the P-cores and part of
  the E-cores; the card that drives the monitors keeps more VRAM free.

The measured effect of each change, and every switch, is documented in
[docs/DUAL_GPU.md](docs/DUAL_GPU.md).

## Requirements

- Two NVIDIA GPUs. The fork was tested on an RTX 5080 16 GB and an RTX 4060 Ti 16 GB.
  A 12 GB + 16 GB pair such as an RTX 5070 and an RTX 5060 Ti also fits the same design;
  start with `--layer-split auto` and adjust the VRAM reserves if one card drives a display
  or shares memory with other programs.
- 32 GB of system RAM. More RAM allows a larger expert cache and a larger context.
- Windows 11 for the ready-made engine. Linux is supported by the source build, but this fork
  was not tested on Linux.
- An NVIDIA driver and CUDA runtime compatible with the engine build.
- Enough disk space for the model files. The recommended model is Qwen3.8-Flash-Next
  Swift 1.5 IQ2_XS; larger quantisations need more VRAM and RAM.

AMD GPUs are not supported by this fork.

## Install on Windows

1. Download this repository, or clone it:

   ```sh
   git clone https://github.com/Hardin22/Strata-DualGPU.git
   cd Strata-DualGPU
   ```

2. Double-click `START-HERE.bat`.
3. Answer the setup questions. With two cards and 32 GB of RAM, choose the low-RAM mode and
   the Swift 1.5 IQ2_XS model.
4. When setup finishes, open the address it prints in a browser, or connect an application to
   the OpenAI-compatible API.

Setup downloads the model and the engine. The ready-made Windows engine is published in the
repository's releases. If a download is interrupted, run `START-HERE.bat` again; it continues
where it stopped.

To use an existing Strata installation, unpack this repository next to it and run
`START-HERE.bat`. Strata shares its data folder between copies on the same PC, so the model is
not downloaded again.

## Install on Linux

```sh
git clone https://github.com/Hardin22/Strata-DualGPU.git
cd Strata-DualGPU
./setup.sh
```

The source build compiles the engine for the local CUDA version. This path was not tested on
Linux by the author of the fork. See [docs/INSTALL.md](docs/INSTALL.md) for the general
requirements and troubleshooting.

## Build from source

The engine is a C++/CUDA program. A source build needs CMake, Ninja or Visual Studio, a CUDA
toolkit, and a C++ compiler.

On Windows, the setup script can build the engine. For a manual build, use the same CMake
options as upstream Strata; the upstream build instructions are in
[README.upstream.md](README.upstream.md) and [docs/INSTALL.md](docs/INSTALL.md).

## Configuration

The setup writes a configuration file named `strata-*.json`. The important settings for a
two-GPU run are:

- `"layer_split": "auto"` lets the server choose the split point and the card order. This is
  the default and is the recommended setting.
- `"gpu_order": "as_given"` keeps the order written in the configuration. Use it only when the
  split point was chosen for that exact order.
- `"vram_reserve": "as_given"` disables the automatic VRAM reserves. The defaults reserve
  1800 MiB on a card that drives a display and 512 MiB on a card that drives none.

The engine flags that control the fork's behaviour are listed in
[docs/DUAL_GPU.md](docs/DUAL_GPU.md#settings-and-switches). The most useful are:

```text
--layer-split auto          choose the split automatically
--pipeline-windows 2        overlap the two stages (default on two GPUs)
--pipeline-windows 0        disable overlapping
--resident-experts          keep the expert copy in system RAM
--no-trim-stage-weights     keep all dense weights on every card
--list-gpus                 print the visible cards and exit
```

Do not expose the server to another computer without an API key. See
[docs/INSTALL.md](docs/INSTALL.md) and [docs/MCP_SERVER.md](docs/MCP_SERVER.md) for the network
options.

## Performance

The numbers below come from the fork's documentation. They were measured on one computer:
RTX 5080 16 GB and RTX 4060 Ti 16 GB, i9-14900KF, 32 GB RAM, Windows 11, Swift 1.5 IQ2_XS,
256K context, Q4_0 KV cache, speculative decoding with the stock draft layer.

| Configuration | Code | Prose | 32K prompt |
| --- | ---: | ---: | ---: |
| Strata 0.1.38 on the RTX 5080 only | 29 tok/s | 28 tok/s | 333 tok/s |
| This fork on both cards | 143 tok/s | 102 tok/s | 1,940 tok/s |

These are measurements on one machine, not guarantees. The speed depends on the cards, the
PCIe layout, the CPU, the model, the context size and the expert cache.

The output is not bit-for-bit identical to upstream. Experts computed on the CPU and experts
computed on the GPU round differently, so a near-tie between two tokens can choose the other
token. The fork's documentation includes a perplexity comparison with upstream.

## Limitations

- Pipelining requires exactly two GPUs. With three or more cards the split still runs, but
  without pipelining.
- The fork is based on Strata 0.1.38. It does not contain later upstream changes.
- The fork was tested on one Windows machine with two NVIDIA cards. Other cards, Linux and
  AMD GPUs were not tested.
- The ready-made draft layer in the releases was trained for Swift 1.5 IQ2_XS. With another
  model it still produces correct answers, but it may be slower than the stock draft layer.
- Like upstream with adaptive expert swaps, the same prompt can produce slightly different
  text from one run to the next at near-ties.

## Upstream, credits and license

This repository is a fork of [Hardin22/Strata-DualGPU](https://github.com/Hardin22/Strata-DualGPU),
which is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).

The Strata engine, server, setup, web application and documentation are the work of Niko1221
and the Strata contributors. The dual-GPU changes and the benchmarks in
[docs/DUAL_GPU.md](docs/DUAL_GPU.md) are from the Strata-DualGPU fork. The upstream README is
kept as [README.upstream.md](README.upstream.md).

Strata is released under the MIT license. See [LICENSE](LICENSE). Model files have their own
licenses; check the model provider's page before redistributing them.
