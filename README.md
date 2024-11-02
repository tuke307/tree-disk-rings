# Tree Ring Analyzer

forked from [hmarichal93/cstrd_ipol](https://github.com/hmarichal93/cstrd_ipol)

## Installation
```bash
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cd ./externas/devernay_1.0 && make clean && make
```

## Usage Examples

### Basic Usage

```bash
tree-ring-analyzer --input IMAGE_PATH --cx CX --cy CY
```

Example:
```bash
tree-ring-analyzer --input input/tree-disk4.png --cx 1204 --cy 1264
```

### Saving Intermediate Results
If you want to save intermediate results, you can use the --save_imgs flag:
```bash
tree-ring-analyzer --input input/tree-disk4.png --cx 1204 --cy 1264 --save_imgs
```

### Using Advanced Parameters
```bash
tree-ring-analyzer --input input/F02c.png --cx 1204 --cy 1264 --output_dir custom_output/ --sigma 4.0 --th_low 10 --th_high 25 --save_imgs --debug
```

## Command-Line Arguments

* `--input` (str, required): Path to input image.
* `--cx` (int, required): Pith x-coordinate.
* `--cy` (int, required): Pith y-coordinate.
* `--output_dir` (str, optional): Output directory path.
* `--root` (str, optional): Root directory of the repository.
* `--sigma` (float, optional): Gaussian kernel parameter for edge detection. Default is 3.0.
* `--th_low` (float, optional): Low threshold for gradient magnitude. Default is 5.0.
* `--th_high` (float, optional): High threshold for gradient magnitude. Default is 20.0.
* `--height` (int, optional): Height after resizing (0 to keep original). Default is 0.
* `--width` (int, optional): Width after resizing (0 to keep original). Default is 0.
* `--alpha` (float, optional): Edge filtering parameter (collinearity threshold). Default is 30.0.
* `--nr` (int, optional): Number of rays. Default is 360.
* `--min_chain_length` (int, optional): Minimum chain length. Default is 2.
* `--debug` (flag, optional): Enable debug mode.
* `--save_imgs` (flag, optional): Save intermediate images.

## Publishing Package
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```
