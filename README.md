# SemiLT
A multi-anchor transfer learning method for cross-modality cell label annotation from scRNA-seq to scATAC-seq.<br>

<img src="https://github.com/Gut2Sdu/SemiLT/blob/main/Supplementary%20Information/Fig-1_00.jpg" width="800px">

## Installation

### Step1. clone repository

```
git clone https://github.com/Gut2Sdu/SemiLT-release.git
```
If you encounter network-related errors (e.g., "Empty reply from server"),
you can alternatively download the complete source code directly from: https://github.com/Gut2Sdu/SemiLT-release
### Step2. create env:
```
conda create -n SemiLT python=3.7 -y
conda activate SemiLT
```

### Step3. install pytorch
```
conda install -y \
  pytorch=1.12.1 \
  torchvision=0.13.1 \
  torchaudio=0.12.1 \
  cudatoolkit=10.2 \
  -c pytorch
```
### Step4. install other dependencies
```
conda install -y \
  numpy=1.21.5 \
  scipy=1.7.3 \
  pandas=1.3.5 \
  scikit-learn=1.0.2 \
  networkx=2.6.3 \
  joblib \
  tqdm \
  -c defaults
```
```
conda install -y \
  anndata=0.8.0 \
  scanpy=1.9.1 \
  -c conda-forge
```

### Step5. Run demo data
After installation, set `DB = 'demo'` in `setting.py` and run `main.py` to verify that SemiLT has been installed successfully.

The example runtime environment: [`environment.txt`](https://github.com/Gut2Sdu/SemiLT-release/blob/main/environment.txt).<br>

## Dataset
All datasets used in our paper can be found.<br>
Data source: <br>

* Data-1,2,3,4: [`RNA+ATAC`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122).<br>

* Data-5: [`RNA+ATAC`](https://github.com/caokai1073/uniPort).<br>

* Data-6: [`RNA`](https://tabula-muris.ds.czbiohub.org/) and [`ATAC`](https://atlas.gs.washington.edu/mouse-atac/).<br>

* Data-7: [`RNA+ATAC`](https://github.com/SydneyBioX/scJoint).<br>

* Data-8: [`RNA`](https://github.com/dpeerlab/Palantir/) and [`ATAC`](https://gitlab.com/cvejic-group/integrative-scrna-scatac-human-foetal).<br>

* Data-9:[`RNA+ATAC`](https://stuartlab.org/signac/articles/pbmc_vignette).

Processed data：
* Baidu Drive (Code:1234):[`Link`](https://pan.baidu.com/s/1TkK4NqDqVRioIoY0P4WNHg?pwd=1234).

## Tutorial
Dimensionality Reduction (e.g., PCA) on scATAC-seq Data.<br>
* Tutorial_PCA: [`data_PCA`](https://github.com/Gut2Sdu/SemiLT-release/blob/main/tutorial/data_PCA.ipynb).<br>

We provide examples to help reproduce our experiments.<br>
* Tutorial1 (Data-5): [`RNA_ATAC`](https://github.com/Gut2Sdu/SemiLT-release/blob/main/tutorial/RNA-seq%20and%20ATAC-seq%20integration%20using%20SemiLT.ipynb).<br>

* Tutorial2 (Data-7): [`CITE_ASAP`](https://github.com/Gut2Sdu/SemiLT-release/blob/main/tutorial/CITE-seq%20and%20ASAP-seq%20integration%20using%20SemiLT.ipynb).<br>

## Reference
Zhitong Chen, Maoteng Duan, Xiaoying Wang, Bingqiang Liu. SemiLT: A Multianchor Transfer Learning Method for Cross-Modality Cell Label Annotation from scRNA-seq to scATAC-seq. *Advanced Science*, 2025. https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202507846
