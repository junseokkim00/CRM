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

3. execute `generate_data.sh`
+ 자신 이름에 맞게 uncomment 후 실행하기
```bash
$ sbatch run.sh
```


4. `estimate_initial_answer.sh`에 자신의 dataset을 넣고, 위의 뽑은 data path 입력 후, 실험 돌리기
5. `prepare_pairwise_data.sh`에서 `initial_response_path`, `crm_data_path`, `dataset` 채우고 돌려서 pairwise_data 만들기
6. To be continue...