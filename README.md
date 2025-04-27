# CRHP
**This is the data and code for our paper** `Collaborative Relation Augmentation with Hierarchical Prescription Inference for Medication Recommendation`.

## Prerequisites

Make sure your local environment has the following installed:


* `pytorch>=1.12.1 & <=1.9`
* `spacy == 2.1.9`
* `tensorboardx == 2.0`
* `tokenizers == 0.7.0`
* `tokenizers == 0.7.0`
* `numpy == 1.15.1`
* `python == 3.7`
* `transformers == 2.9.1`

## Datastes

We provide the dataset in the [datas](datas/) folder.

| Data      | Source                                                   | Description                                                  |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| MIMIC-III | [This link](https://physionet.org/content/mimiciii/1.4/) | MIMIC-III is freely-available database from 2001 to 2012, which is associated with over forty thousand patients who stayed in critical care units |
| MIMIC-IV  | [This link](https://physionet.org/content/mimiciv/2.2/)  | MIMIC-IV is freely-available database between 2008 - 2019, which is associated with 299,712 patients who stayed in critical care units |

## Documentation


* src
    * README.md
    * data_loader.py
    * train.py
    * model_net.py
    * outer_models.py
    * util.py
    * CKG.py
    * processing.py
  
* datas
    * origin
        * drug-atc.csv: drug to atc code mapping file.
        * ndc2atc_level4.csv: NDC to RXCUI mapping file.
        * ndc2rxnorm_mapping.txt: NDC to RXCUI mapping file.
        * drug-DDI.csv: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
    * output
        * ddi_A_final.pkl：ddi adjacency matrix
        * ehr_adj_final.pkl：: if two drugs appear in one set, then they are connected
        * records_final.pkl: the final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split on MIMIC_III dataset
        * records_final_4.pkl: the final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split on MIMIC_IV dataset
        * voc_final.pkl：diag/pro/med index to code dictionary on MIMIC_III dataset
        * voc_final_4.pkl: diag/pro/med index to code dictionary on MIMIC_IV dataset
    * 



 After the paper is accepted, we will further upload the relevant data preprocessing files.

## Step 1: Data Processing 

* Download the MIMIC-III/MIMIC-IV dataset from [MIMIC-III link](https://physionet.org/content/mimiciii/1.4/) / [MIMIC-IV link](https://physionet.org/content/mimiciv/1.4/)

* Extract three main files(PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz), and change the path comply with current state:
```
# med_file = '/data/mimic-iii/PRESCRIPTIONS.csv'
# diag_file = '/data/mimic-iii/DIAGNOSES_ICD.csv'
# procedure_file = '/data/mimic-iii/PROCEDURES_ICD.csv'
```
## Step 2: Package Dependency
First, install the [conda](https://www.anaconda.com/)
Then, create the conda environment through yaml file:

```
conda env create -f mrln_env.yaml
```

## Step 3: run the code
```
python train.py
```
Please run `train.py` to begin training and testing.

On a single NVIDIA® GeForce RTX™ 3080 Ti (10GB) GPU, a typical run takes 2.5 hours to complete.

## TODO
More training scripts for easy training will be added soon.

