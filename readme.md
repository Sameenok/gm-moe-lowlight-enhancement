
<div align="center">
<p align="center"> <img src="figure/logo.png" width="200px"> </p>

# GM-MoE :      Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts


[[`Arxiv`](https://arxiv.org/abs/2503.07417)] [[`Github`](https://github.com/Sameenok/gm-moe-lowlight-enhancement.git)]




&nbsp;

## 1. Environment Setup

We recommend the following environment (tested on 4x RTX 4090 GPUs):

```bash
# Python and CUDA version
python 3.9.5
CUDA 11.3

# Clone the repository and enter the project directory
cd GM-MoE

# Install Python dependencies
pip install -r requirements.txt
pip install setproctitle timm
pip install numpy==1.23.5

# (Recommended) Install PyTorch and related packages
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# Install the project in development mode (without CUDA extensions)
python setup.py develop --no_cuda_ext
```

**Note:**
- The environment was tested on 4x NVIDIA RTX 4090 GPUs (conda environment name: `rgt`).
- Adjust CUDA and PyTorch versions as needed for your hardware.


## 2. Prepare Dataset

Download the following datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)


<details close>
<summary><b> Then organize these datasets as follows: </b></summary>

```
datasets/
└── LOLv1v2/
    ├── LOLv1/
    │   ├── our485/
    │   │   ├── low/
    │   │   └── high/
    │   └── eval15/
    │       ├── low/
    │       └── high/
    └── LOLv2/
        ├── Synthetic/
        │   ├── Train/
        │   │   ├── Low/
        │   │   └── Normal/
        │   └── Test/
        │       ├── Low/
        │       └── Normal/
        └── Real_captured/
            ├── Train/
            │   ├── Low/
            │   └── Normal/
            └── Test/
                ├── Low/
                └── Normal/

```

</details>

We also provide download links for LIME, NPE, MEF, DICM, and VV datasets that have no ground truth:

[Baidu Disk](https://pan.baidu.com/s/1oHg03tOfWWLp4q1R6rlzww?pwd=cyh2) (code: `cyh2`)
 or [Google Drive](https://drive.google.com/drive/folders/1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp?usp=drive_link)


&nbsp;                    

## 3. Testing

### Results and Pre-trained Models

| Model Name                  | Dataset         | PSNR  | SSIM  | Pretrained Models                                                                    | Config Files                                                                                                                    |
| :-------------------------- | :-------------- | :---- | :---- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| GMMoE-LOLv1                | LOLv1           | 26.66 | 0.857 | [Baidu Drive](https://pan.baidu.com/s/1yDuAnjwMPugVKgFaoqd_cQ?pwd=xgph) Code: `xgph` | [Train](./options/train/OURNet_4_GPU/train_LOLv1.yml) \| [Test](./options/test/OURNet_4_GPU/test_LOLv1.yml)                     |
| GMMoE-LOLv2-Real\_captured | LOLv2-Real      | 23.65 | 0.806 | [Baidu Drive](https://pan.baidu.com/s/1uCNpl5uC2g4xMwps9HgPkw?pwd=4t2v) Code: `4t2v` | [Train](./options/train/OURNet_4_GPU/train_LOLv2_Real_captured.yml) \| [Test](./options/test/OURNet_4_GPU/test_LOLv2_Real.yml)  |
| GMMoE-LOLv2-Synthetic      | LOLv2-Synthetic | 26.30 | 0.937 | [Baidu Drive](https://pan.baidu.com/s/1QPlsIlRoM60Q83P6yCrZYQ?pwd=4far) Code: `4far` | [Train](./options/train/OURNet_4_GPU/train_LOLv2_Synthetic.yml) \| [Test](./options/test/OURNet_4_GPU/test_LOLv2_Synthetic.yml) |

---

### Batch Testing (Entire Dataset)
Download our models from [Baidu Disk](https://pan.baidu.com/s/1HaQxPFfVftI5fA5nAdUkXA?pwd=znjp) (code: `znjp`) . Put them in folder `experiments`

```bash
# LOL-v1
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/OURNet_4_GPU/test_LOLv1.yml --launcher pytorch

# LOL-v2-real
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 basicsr/test.py -opt options/test/OURNet_4_GPU/test_LOLv2_Real.yml --launcher pytorch

# LOL-v2-synthetic
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4323 basicsr/test.py -opt options/test/OURNet_4_GPU/test_LOLv2_Synthetic.yml --launcher pytorch
```

**Notes:**

* `--nproc_per_node=1`: Specifies single-GPU testing. Adjust the value for multi-GPU testing.
* `--master_port`: Set a custom port to avoid conflicts.

---

### Single Image Testing

```bash
# LOL-v1
python basicsr/demo.py -opt options/test/OURNet_4_GPU/test_LOLv1.yml --input_path ./demo/your_input.png --output_path ./demo/your_output.png

# LOL-v2-real
python basicsr/demo.py -opt options/test/OURNet_4_GPU/test_LOLv2_Real.yml --input_path ./demo/your_input.png --output_path ./demo/your_output.png

# LOL-v2-synthetic
python basicsr/demo.py -opt options/test/OURNet_4_GPU/test_LOLv2_Synthetic.yml --input_path ./demo/your_input.png --output_path ./demo/your_output.png
```

**Parameters:**

* `--input_path`: Path to the low-light input image.
* `--output_path`: Path where the enhanced image will be saved.




&nbsp;


## 4. Training

You can check our training logs at [Baidu Drive](https://pan.baidu.com/s/1jhPzR4agXtZaJVR4WPBCsA?pwd=beum) (code: `beum`) .

We recommend using a PyTorch 1.11 environment to train our models.

```shell
# Activate your environment
conda activate your_env_name

# LOL-v1
python basicsr/train.py -opt options/train/OURNet_4_GPU/train_LOLv1.yml

# LOL-v2-real
python basicsr/train.py -opt options/train/OURNet_4_GPU/train_LOLv2_Real_captured.yml

# LOL-v2-synthetic
python basicsr/train.py -opt options/train/OURNet_4_GPU/train_LOLv2_Synthetic.yml
```

For distributed multi-GPU training (recommended), use the following commands:

```shell
# LOL-v1
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/OURNet_4_GPU/train_LOLv1.yml --launcher pytorch

# LOL-v2-real
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4322 basicsr/train.py -opt options/train/OURNet_4_GPU/train_LOLv2_Real_captured.yml --launcher pytorch

# LOL-v2-synthetic
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4323 basicsr/train.py -opt options/train/OURNet_4_GPU/train_LOLv2_Synthetic.yml --launcher pytorch
```

**Notes:**
- `--nproc_per_node=4` means using 4 GPUs, adjust according to your hardware.
- `--master_port` can be set to any available port to avoid conflicts.
- Training logs and model weights will be saved automatically in the `experiments/` directory.


&nbsp;


## 5. Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{liao2025gmmoelowlightenhancementgatedmechanism,
      title={GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts}, 
      author={Minwen Liao and Hao Bo Dong and Xinyi Wang and Kurban Ubul and Ziyang Yan and Yihua Shao},
      year={2025},
      eprint={2503.07417},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07417}, 
}
