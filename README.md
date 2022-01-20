# Regularized Softmax Deep Multi-Agent Q-Learning

This repository is the implementation of [Regularized Softmax Deep Multi-Agent Q-Learning](https://proceedings.neurips.cc/paper/2021/file/0a113ef6b61820daa5611c870ed8d5ee-Paper.pdf) in NeurIPS 2021. This codebase is based on the open-source [PyMARL](https://github.com/oxwhirl/pymarl) framework and [maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch), and please refer to that repo for more documentation.

## Citing
If you used this code in your research or found it helpful, please consider citing our paper:
Bibtex:
```
@inproceedings{pan2021regularized,
  title={Regularized Softmax Deep Multi-Agent Q-Learning},
  author={Pan, Ling and Rashid, Tabish and Peng, Bei and Huang, Longbo and Whiteson, Shimon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Requirements
- PyMARL: Please check the [PyMARL](https://github.com/oxwhirl/pymarl) repo for more details about the environment.
- Multi-agent Particle Environments: in envs/multiagent-particle-envs and install it by `pip install -e .`
- SMAC: Please check the [SMAC](https://github.com/oxwhirl/smac) repo for more details about the environment. Note that for all SMAC experiments we used the latest version SC2.4.10. The results reported in the SMAC paper (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232. Performance is not always comparable across versions.

## Usage
Please follow the instructions below to replicate the results in the paper. Hyperparameters can be found in config files and main.py for multi-agent particle environments including predator-prey, physical deception, world, and communication, and SMAC environments including 2s3z, 3s5z, 2c_vs_64zg, and MMM2.

### Multi-Agent Particle Environments
```
python3 src/main.py --config=res_qmix --env-config=mpe_env with scenario_name=<SCENARIO_NAME> seed=<SEED>
```

### StarCraft Multi-Agent Challenge
```
bash run.sh <GPU> python3 src/main.py --config=res_qmix --env-config=sc2 with env_args.map_name=<MAP_NAME> seed=<SEED>
```
