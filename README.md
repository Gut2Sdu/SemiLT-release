# SemiLT
SemiLT: a multi-anchor transfer learning method for cross-modality cell label annotation from scRNA-seq to scATAC-seq.<br>

<img src="https://github.com/Gut2Sdu/SemiLT/blob/main/Supplementary%20Information/Fig-1_00.jpg" width="800px">

## Installation

SemiLT can be obtained by simply clonning the github repository:

```
git clone https://github.com/Gut2Sdu/SemiLT-release.git
```

The following python packages are required to be installed before running SemiLT: 
`scanpy`, `torch`, `itertools`, `scipy`, `numpy` and `sklearn`.

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

## Tutorial
We provide examples to help reproduce our experiments.<br>
* Tutorial1 (Data-5): [`RNA_ATAC`](https://github.com/Gut2Sdu/SemiLT-release/blob/main/tutorial/RNA-seq%20and%20ATAC-seq%20integration%20using%20SemiLT.ipynb).<br>

* Tutorial2 (Data-7): [`CITE_ASAP`](https://github.com/Gut2Sdu/SemiLT-release/blob/main/tutorial/CITE-seq%20and%20ASAP-seq%20integration%20using%20SemiLT.ipynb).<br>
