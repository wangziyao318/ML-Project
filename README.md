# ML Project

We try to segment cardiac MRI 3D images from [Medical Decathlon](http://medicaldecathlon.com/) using [nnUNet](https://github.com/MIC-DKFZ/nnUNet/) that performed well in its competition, and use dice / NSD metrics to evaluate its accuracy from test set.

## Procedures

### Following instructions to deploy nnUNet

Create venv (or using vscode function)

```sh
python -m venv .venv
```

In the virtual environment, install torch

```sh
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

Install nnUNet as pip package

```sh
pip3 install -e path/to/nnUNet # modified
pip3 install nnunetv2 # original
```

A sample dataset.json

```json
{
  "channel_names": {
    "0": "MRI"
  },
  "labels": {
    "background": 0,
    "Anterior": 1,
    "Posterior": 2
  },
  "numTraining": 260,
  "numTest": 130,
  "file_ending": ".nii.gz",
  "overwrite_image_reader_writer": "SimpleITKIO"
}
```

### Train nnUNet on imagesTr given labelsTr

- we use dataset004_hippocampus
- training and test data are each half of whole training set, we discard test set since it has no labels data

preprocess, plan and generate fingerprint

```sh
nnUNetv2_plan_and_preprocess -d 4 -c 3d_fullres
# 10 threads, use 3d_fullres configuration for small size image, on number 1 dataset
```
train: training needs 1000 epoches for each fold in 5-fold cross validation

```sh
nnUNetv2_train 4 3d_fullres 0 -device cuda
# use cuda, on number 1 dataset, use 3d_fullres configuration, fold 0 in 5-fold cross validation
```

### Use trained nnUNet to segment imagesTs

predict based on average of all folds 0-4, using checkpoint final and best

```sh
nnUNetv2_predict -i .\raw\Dataset004_Hippocampus\imagesTs\ -o .\results\Dataset004_Hippocampus\nnUNetTrainer__nnUNetPlans__3d_fullres\pred\ -d 4 -c 3d_fullres -f 0 -chk checkpoint_final.pth -npp 4 -nps 4 -device cuda
# use half data model to predict the other half
```

evaluate predict accuracy (preprocess test set first to generate gt segments)

```sh
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity -c 3d_fullres -np 10
```

```sh
nnUNetv2_evaluate_folder -djfile .\nnUNet_preprocessed\Dataset002_Hippocampus\dataset.json -pfile .\nnUNet_preprocessed\Dataset002_Hippocampus\nnUNetPlans.json .\nnUNet_preprocessed\Dataset002_Hippocampus\gt_segmentations\ .\nnUNet_results\Dataset001_Hippocampus\nnUNetTrainer__nnUNetPlans__3d_fullres\pred\
```

## Optimization

1. dataset.json channel_names zscore specify to MRI to customize normalization https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md#how-to-implement-custom-normalization-strategies
2. also use region-based training with "regions_class_order" adding to dataset.json https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
3. hiddenlayer to generate neural network structure https://github.com/waleedka/hiddenlayer
4. set "nnUNet_n_proc_DA" training threads
5. change 5-fold cross validation split https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
6. 