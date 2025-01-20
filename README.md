# SemiLT
SemiLT: a semi-supervised transfer learning method transfers cell labels from scRNA-seq to scATAC-seq.<br>
Code reference [`scJoint`](https://github.com/SydneyBioX/scJoint).<br>
<img src="https://github.com/Gut2Sdu/SemiLT/blob/main/Supplementary%20Information/Fig-1_00.jpg" width="800px">

## Installation
The following python packages are required to be installed before running SemiLT: 
`scanpy`, `torch`, `itertools`, `scipy`, `numpy` and `sklearn`.

## Dataset
All datasets used in our paper can be found.<br>
Data source: <br>

* Data-1,2,3,4: [`RNA+ATAC`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122).<br>

* Data-5: [`RNA+ATAC`](https://github.com/caokai1073/uniPort).<br>

* Data-6: [`RNA`](https://tabula-muris.ds.czbiohub.org/) and [`ATAC`](https://atlas.gs.washington.edu/mouse-atac/).<br>

* Data-7: [`RNA+ATAC`](https://github.com/SydneyBioX/scJoint).<br>

* Data-8: [`RNA+ATAC`](https://github.com/dpeerlab/Palantir/).<br>

* Data-9:[`RNA+ATAC`](https://stuartlab.org/signac/articles/pbmc_vignette).

## Tutorial
We provide examples to help reproduce our experiments.<br>
* Tutorial1: [`CITE-ASAP`](https://github.com/Gut2Sdu/SemiLT/blob/main/tutorial/CITE-seq%20and%20ASAP-seq%20integration%20using%20SemiLT.ipynb).<br>

* Tutorial2: [`MCA-subset`](https://github.com/Gut2Sdu/SemiLT/blob/main/tutorial/MCA-subset%20dataset%20integration%20using%20SemiLT.ipynb).<br>
