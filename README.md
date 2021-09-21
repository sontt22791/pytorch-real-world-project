# pytorch-real-world-project

## Chapter 01: Getting Started with PyTorch

- 1 số lưu ý:
    - việc chuyển đổi qua lại giữa numpy và tensor ko phát sinh `cost on the performance of your app` vì NumPy and PyTorch store data in memory in the same way
    - Almost all operations have an in-place version - the name of the operation, followed by an `underscore` (`_`), vd `add_`, `unsqueeze_`,...

## Chapter 02: Build Your First Neural Network with PyTorch

- `nn.BCELoss`:  It expects the values to be outputed by the `sigmoid function`.
- `nn.BCEWithLogitsLoss` => This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.