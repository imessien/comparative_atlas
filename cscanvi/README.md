# cSCANVI (fork)

Vendored fork of [theislab/comparative_atlas](https://github.com/theislab/comparative_atlas) `cscanvi/`,
maintained in [imessien/Inflammaging_Network](https://github.com/imessien/Inflammaging_Network).

Upstream divergences:

- scvi-tools **1.3.3** + Lightning 2.x DDP training
- Energy-only test-time adaptation (`cscanvi._adapt.TTA_SCANVI`)
- Backed Parse 10M pipeline: CuPy HVG, `BackedScanviDataModule`, gene-panel model slicing
- `data/` package: unified gene panel + PyTorch streaming loaders

Not a git submodule — this tree is the canonical copy for this project.
