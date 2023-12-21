# ML Project

We try to segment cardiac MRI 3D images from [Medical Decathlon](http://medicaldecathlon.com/) using [nnUNet](https://github.com/MIC-DKFZ/nnUNet/) that performed well in its competition, and use dice / NSD metrics to evaluate its accuracy from test set.

## Procedures

### Following instructions to deploy nnUNet


### Train nnUNet on imagesTr given labelsTr

- dataset hippocampus is 4 times faster than cardiac, due to small image size
- dataset 1 and 2 are half of training set, we discard test set since it has no labels data

preprocess, plan and generate fingerprint

```sh
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres -np 8
# 8 threads, use 3d_fullres configuration for small size image, on number 1 dataset, integrity verified
```
train: training needs 1000 epoches for each fold in 5-fold cross validation

```sh
nnUNetv2_train 1 3d_fullres ? -device cuda --npz
# use cuda, on number 1 dataset, use 3d_fullres configuration, fold 0-4 in 5-fold cross validation, npz save softmax pred in final validation
```

### Use trained nnUNet to segment imagesTs

predict based on average of all folds 0-4

```sh
nnUNetv2_predict -i .\nnUNet_raw\Dataset002_Hippocampus\imagesTr\ -o .\nnUNet_results\Dataset001_Hippocampus\nnUNetTrainer__nnUNetPlans__3d_fullres\pred\ -d 1 -c 3d_fullres -f ? -chk checkpoint_final.pth -npp 4 -nps 4 -device cuda
# use half data model to predict the other half
```

## Optimization

1. dataset.json channel_names zscore specify to MRI to customize normalization https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md#how-to-implement-custom-normalization-strategies
2. also use region-based training with "regions_class_order" adding to dataset.json https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
3. hiddenlayer to generate neural network structure https://github.com/waleedka/hiddenlayer
4. set "nnUNet_n_proc_DA" training threads
5. change 5-fold cross validation split https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
6. 