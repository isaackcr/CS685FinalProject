# CS685FinalProject

## Configure Python Environment
1. Install conda
   - https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
1. Create and activate conda environment
    ```
    conda init
    conda create -n CS685FinalProject
    conda activate CS685FinalProject
    ```
1. Install Dependencies
    ```
    conda install -y -c anaconda python=3.9 scikit-learn nbconvert jinja2==3.0.3
    conda install -y -c conda-forge evaluate matplotlib pyppeteer
    conda install -y -c huggingface transformers datasets
    pip install torch torchvision torchaudio
    ```
1. If Using CUDA, Install GPU support
   - Install drivers: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
   - Install Python CUDA support
    ```
    conda install -y -c conda-forge cudatoolkit
    ```
## Download raw data 
- you must have username/password on physionet.org with access to the MIMIC IV dataset
```
mkdir -p data
export MIMIC_USER=
export MIMIC_PW=
wget -P data -N -c -r -np --user $MIMIC_USER --password $MIMIC_PW https://physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz
```
