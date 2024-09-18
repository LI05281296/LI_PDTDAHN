# PDTDAHN
This repository provides the data and codes for the model PDTDAHN, which can predict drug-target-disease triple associations. A heterogeneous network is constructed, which contains drugs, target proteins, and diseases as nodes and six association types (drug-disease, drug-target, disease-target, drug-drug, target-target, disease-disease associations) as edges. The network embedding algorithm, Mashup, was adopted to extract feature representations of drugs, targets, and diseases from this network. The LightGBM is adopted as the prediction engine. 
![image](https://github.com/LI05281296/PDTDAHN/blob/main/Framework.png)
# Requirements
      python 3.9.15
      pandas 2.1.4
      numpy 1.24.3
      scipy 1.12.0
      scikit-learn 1.4.1
# Directory Structure Overview

`data`
This folder contains the data used by the research institute

`mashup`
This folder contains feature extraction algorithm：mashup

`src`
This folder contains Python files for various classification prediction algorithms

`DTINET`
This folder contains comparative experiments by using DTINET

`LBMFF`
This folder contains comparative experiments by using LBMFF


