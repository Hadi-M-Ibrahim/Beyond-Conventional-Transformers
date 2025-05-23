# Beyond Conventional Transformers

## Directory Structure

```
SupplementalMaterials/
├── CODE_OF_CONDUCT.md
├── LICENSE
├── README.md (You are here)
├── ImprovedBranch/
│   ├── requirements.txt (Pip install this to run the code; see instructions below)
│   ├── EfficientViT/
│   │   ├── engine.py (Look here for the training loop)
│   │   ├── losses.py (Look here for KD functionality)
│   │   ├── main.py (Run this to train the model; see instructions below)
│   │   ├── attention_map_visualizer.py (Run this to recreate attention maps; see instructions below)
│   │   ├── MXA_visualizer.py (Run this to recreate ROI visualization; see instructions below)
│   │   ├── test.py (Run this to eval model checkpoints on test set [see links]; see instructions below)
│   │   ├── utils.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py (Look here for the dataset class and data augmentations)
│   │   │   ├── samplers.py
│   │   │   ├── threeaugment.py
│   │   ├── models/
│   │       ├── __init__.py
│   │       ├── build.py (Look here for model variants)
│   │       ├── efficientvit.py (Look here for the model architecture w/ the MXA)
├── NaiveBranch/
│   ├── requirements.txt (Pip install this to run the code; see instructions below)
│   ├── EfficientViT/
│   │   ├── engine.py (Look here for the training loop)
│   │   ├── losses.py (Look here for KD functionality)
│   │   ├── main.py (Run this to train the model; see instructions below)
│   │   ├── test.py (Run this to eval model checkpoints on test set [see links]; see instructions below)
│   │   ├── utils.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py (Look here for the dataset class and data augmentations)
│   │   │   ├── samplers.py
│   │   │   ├── threeaugment.py
│   │   ├── model/
│   │       ├── __init__.py
│   │       ├── build.py (Look here for model variants)
│   │       ├── efficientvit.py (Look here for the model architecture w/o the MXA)
├── Results/
│   ├── ablation_study/
│   │   ├── augs:u1/
│   │   │   ├── args.txt (Look here for hyperparameters used for training)
│   │   │   ├── log.txt (Look here for validation set logs)
│   │   │   ├── model.txt (Look here for model spesfications)
│   │   ├── augs:u1+MXA/
│   │   │   ├── args.txt (Look here for hyperparameters used for training)
│   │   │   ├── log.txt (Look here for validation set logs)
│   │   │   ├── model.txt (Look here for model spesfications)
│   │   ├── augs:u1+MXA+KD/
│   │       ├── args.txt (Look here for hyperparameters used for training)
│   │       ├── log.txt (Look here for validation set logs)
│   │       ├── model.txt (Look here for model spesfications)
│   ├── Improved/
│   │   ├── TestSet/
│   │   │   ├── Run0/
│   │   │   │   ├── log.txt (Look here for test set logs)
│   │   │   ├── Run1/
│   │   │   │   ├── log.txt (Look here for test set logs)
│   │   │   ├── Run2/
│   │   │       ├── log.txt (Look here for test set logs)
│   │   ├── ValSet/
│   │       ├── args.txt (Look here for hyperparameters used for training)
│   │       ├── log.txt (Look here for validation set logs)
│   │       ├── model.txt (Look here for Model spesfications)
│   ├── Naive/
│       ├── TestSet/
│       │   ├── Run0/
│       │   │   ├── log.txt (Look here for test set logs)
│       │   ├── Run1/
│       │   │   ├── log.txt (Look here for test set logs)
│       │   ├── Run2/
│       │       ├── log.txt (Look here for test set logs)
│       ├── ValSet/
│           ├── args.txt (Look here for hyperparameters used for training)
│           ├── log.txt (Look here for validation set logs)
│           ├── model.txt (Look here for Model spesfications)
```
## Instructions (Linux Strongly Recomended)
### ImprovedBranch
1. **Install Requirements**: 
   ```bash
   pip install -r ./ImprovedBranch/requirements.txt
   ```
2. **main.py (train model)**: 
   ```bash
   cd ./ImprovedBranch/EfficientViT
   ```
   ```bash
    python main.py --model EfficientViT_MultiLabel_M5 --data-set CHEXPERT --data-path "(directory of CheXpert folder)" --batch-size 512 --epochs 50 --output_dir "(select an output directory)" --device cuda --teacher-model densenet121 --distillation-type soft --num_workers "(start with # of cpu cores) or zero for windows not recommended"
   ```
3. **test.py (run test set)**: 
   ```bash
   cd ./ImprovedBranch/EfficientViT
   ```
   ```bash
    python test.py --model EfficientViT_MultiLabel_M5 --data-path "(directory of CheXpert folder containg test.csv [see links])" --checkpoint-dir "(directory containing model checkpoints [see links])" --output-dir "(select an output directory)" --num-workers "(start with # of cpu cores) or zero for windows not recommended" 
   ```
4. **MXA_visualizer.py (recreate ROI visualization like in apendix F)**: 
   ```bash
    cd ./ImprovedBranch/EfficientViT
    ```
    ```bash
      python MXA_visualizer.py -image-path "(path to CXR)" --output "(select an output dir)"
    ```
5. **attention_map_visualizer.py (recreate attention maps like in appendix G)**:
   ```bash
    cd ./ImprovedBranch/EfficientViT
    ```
    ```bash
      python attention_map_visualizer.py --naive_checkpoint "(path to naive checkpoint [see links])" --improved_checkpoint"(path to improved checkpoint [see links]" --image_dir "(path to CXR)" --output_dir "(select an output dir)"
    ```

### NaiveBranch
1. **Install Requirements**: 
   ```bash
   pip install -r ./NaiveBranch/requirements.txt
   ```
2. **main.py (train model)**: 
   ```bash
   cd ./NaiveBranch/EfficientViT
   ```
   ```bash
    python main.py --model EfficientViT_MultiLabel_M5 --data-set CHEXPERT  --data-path "(directory of CheXpert folder)" --batch-size 512 --epochs 50 --output_dir "(select an output directory)" --device cuda --distillation-type none --num_workers "(start with # of cpu cores) including windows"
   ```
3. **test.py (run test set)**: 
   ```bash
   cd ./NaiveBranch/EfficientViT
   ```
   ```bash
    python test.py --model EfficientViT_MultiLabel_M5 --data-path "(directory of CheXpert folder containg test.csv [see links])" --checkpoint-dir "(directory containing model checkpoints [see links])" --output-dir"(select an output directory)" --num-workers "(start with # of cpu cores) including windows"
   ```

## Links

- **Selected Checkpoints**: *[https://drive.google.com/file/d/1g7EjpzGraGzJr0ewOV2nA-oPooix4wiS/view?usp=sharing](https://drive.google.com/file/d/1g7EjpzGraGzJr0ewOV2nA-oPooix4wiS/view?usp=sharing)*
- **CheXpert Dataset**: *[https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)*
- **CheXpert Test Labels**: *[https://github.com/rajpurkarlab/cheXpert-test-set-labels](https://github.com/rajpurkarlab/cheXpert-test-set-labels)*
- **Additional (& Most Up To Date) Supplemental Files**: *[https://drive.google.com/drive/folders/1AfXz9uF-PHm0HbwvoTSJufilwSdGOjof?usp=sharing](https://drive.google.com/drive/folders/1AfXz9uF-PHm0HbwvoTSJufilwSdGOjof?usp=sharing)*
