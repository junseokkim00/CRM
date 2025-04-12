# CRM

## Instructions

1. Setup python environment
```bash
$ conda create -n crm python=3.10
$ conda activate crm
$ conda install ipykernel
$ pip install -r requirements.txt
```

2. enter `HF_TOKEN` at `.env`
+ refer to [this link](https://huggingface.co/docs/hub/security-tokens)

3. execute `run.sh`
```bash
$ sbatch run.sh
```