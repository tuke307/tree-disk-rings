# CS-TRD: a Cross Sections Tree Ring Detection method
Repository for the IPOL paper "CS-TRD: a Cross Sections Tree Ring Detection method". Submitted on 13/05/2023. Last Revision on 19/11/2023. 
IPOL Demo: [IPOL][link_ipol_paper].
UruDendro ImageSet: [UruDendro][link_urudendro].
ArXiv paper: [ArXiv][link_arxiv_paper].

[link_ipol_paper]: https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000390
[link_urudendro]: https://iie.fing.edu.uy/proyectos/madera/
[link_arxiv_paper]: https://doi.org/10.48550/arXiv.2305.10809

![F03d_compare.jpg](assets%2FF03d_compare.jpg)



Version 1.0
Last update: 13/05/2023
Authors: 
-	Henry Marichal, henry.marichal@fing.edu.uy
-   Diego Passarella, diego.passarella@cut.edu.uy
-   Gregory Randall, randall@fing.edu.uy

## Get started

#### 1. Folders
All the python source files are in lib/ folder. Dependencies are in externas/ folder. 
Default configuration files are in config/ folder. 


#### Paper    
Algorithm 1 is implemented in the file **main.py**

Canny-Devernay implementation [[10](https://www.ipol.im/pub/art/2017/216/)] is in **externas/devernay_1.0**. Python
wrapper is in **lib/canny_devernay_edge_detector.py**

Algorithm 2, 3 and 4 are implemented in the file **lib/connect_chains.py**

#### Supplementary Material

Algorithm 1, 2 and 3 are implemented in the file **lib/preprocessing.py**

Algorithm 4 and 5 are implemented in the file **lib/filter_edges.py** 

Algorithm 6 is implemented in the file **lib/sampling.py**

Algorithm 7, 8, 9, 10,  12, 13, 14 are implemented in the file **lib/connect_chains.py**

Algorithm 11 is implemented in the file **lib/basic_properties.py**

Algorithm 15, 16, 17 and 18 are implemented in the file **lib/postprocessing.py**

## Installation
```bash
conda create --name ipol
conda activate ipol
conda install -n ipol -c conda-forge geos
conda install -n ipol -c anaconda cmake 
conda install -n ipol pip
```
```bash
pip install --no-cache-dir -r requirements.txt
```
```bash
cd ./externas/devernay_1.0 && make clean && make
```

## Examples of usage

Here some examples of usage:
```bash
python main.py --input IMAGE_PATH --cx CX --cy CY 
  --outpu_dir OUTPUT_DIR --root REPO_ROOT_DIR
```
```bash
python main.py --input input/F02c.png --cy 1264 --cx 1204  --output_dir ./output --root ./
```
If you want to run the algorithm generating intermediate results you can use the flag --save_imgs

```bash
python main.py --input input/F02c.png --cy 1264 --cx 1204  --output_dir ./output --root ./ --save_imgs 1
```

## Automatic center detection
Detecting pith center automatically can be done using software from IPOL paper "Ant Colony Optimization for Estimating Pith Position on Images of Tree Log Ends" [IPOL][link_ipol_pith_paper].

[link_ipol_pith_paper]: https://www.ipol.im/pub/art/2022/338/?utm_source=doi

## Segmentation with U-Net
Salient object segmentation is done using U-Net. The model should be downloaded to the input folder.

```bash
python salient_object_segmentation.py --input IMAGE_PATH --output OUTPUT_PATH --model ./input/u2net.pth
```

## Docker Container
You can run the algorithm in a docker container.

### Pull the image
```bash
docker pull hmarichal/cstrd:v1.0
```

### Run the container
In order to run the container you need to mount a volume with the data you want to process (YOUR_DATA_FOLDER). Results 
will be stored in the mounted volume. Run the following command:
```bash
docker run -v YOUR_DATA_FOLDER:/workdir/bin/output -it hmarichal/cstrd:v1.0 / 
 python main.py --input YOUR_DATA_FOLDER/image_path --cy 1264 --cx 1204 /
 --output_dir ./output --root ./ --save_imgs 1
```

### Build the image
```bash
 docker build -f .ipol/Dockerfile . -t hmarichal/cstrd:v1.0
```

## Citation
```
@misc{marichal2023cstrd,
      title={CS-TRD: a Cross Sections Tree Ring Detection method}, 
      author={Henry Marichal and Diego Passarella and Gregory Randall},
      year={2023},
      eprint={2305.10809},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
License for th source code: [MIT](./LICENSE)





