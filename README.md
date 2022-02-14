# Efficient Guided Evolution for Neural Architecture Search


## Usage 

Create a conda environment using the env.yml file

```bash
conda env create -f env.yml
```

Activate the environment and follow the instructions to install
```
conda activate gea
```

Install nasbench (see https://github.com/google-research/nasbench)

Download the NDS data from https://github.com/facebookresearch/nds and place the json files in path_to_code/nds_data/
Download the NASbench101 data (see https://github.com/google-research/nasbench)
Download the NASbench201 data (see https://github.com/D-X-Y/NAS-Bench-201)

Reproduce all of the results by running 

```bash
./run.sh
```

The code is licensed under the MIT licence.


## Acknowledgements

This repository makes liberal use of code from the [AutoDL](https://github.com/D-X-Y/AutoDL-Projects) library, [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201), [NAS-Bench-101](https://github.com/google-research/nasbench) and [NAS-WOT](https://github.com/BayesWatch/nas-without-training). We are grateful to the authors for making the implementations publicly available.


## Citing us

If you use or build on our work, please consider citing us:

```bibtex
@inproceedings{gea2021,
    title={Guided Evolution for Neural Architecture Search},
    author={Vasco Lopes and Miguel Santos and Bruno Degardin and Luís A Alexandre},
    year={2021},
    booktitle={Advances in Neural Information Processing Systems 35 (NeurIPS) - New In ML}
}
```

