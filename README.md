# ELEC523 Project

We try to segment cardiac MRI 3D images from [Medical Decathlon](http://medicaldecathlon.com/) using [nnUNet](https://github.com/MIC-DKFZ/nnUNet/) that performed well in its competition, and use dice / NSD metrics to evaluate its accuracy from test set.

## Procedures

### Following instructions to deploy nnUNet


### Train nnUNet on imagesTr given labelsTr

> dataset hippocampus is 4 times faster than cardiac, due to small image size
> dataset 1 is original, 2 and 3 are half of 1 for fine-tuning

1. preprocess, plan and generate fingerprint
```sh
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres -np 8
# 8 threads, use 3d_fullres configuration, on number 1 dataset, integrity verified
```
2. train: since final train needs 1000 epoches, we omit it and only use 150 for similar accuracy
```sh
nnUNetv2_train 1 3d_fullres all -device cuda
# use cuda, on number 1 dataset, use 3d_fullres configuration, fold all means do not cross validate (default is 5-fold cross validation)
```

### Use trained nnUNet to segment imagesTs

1. predict

```sh
nnUNetv2_predict -i .\nnUNet_raw\Dataset003_Hippocampus\imagesTr\ -o .\nnUNet_results\Dataset002_Hippocampus\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all_pred_final\ -d 2 -c 3d_fullres -f all -chk checkpoint_final.pth -npp 4 -nps 4 -device cuda
```
2.evaluate only based on training data, see [summary.json](./nnUNet_results/Dataset002_Hippocampus/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all_pred_final/summary.json)

```sh
nnUNetv2_evaluate_folder -djfile .\nnUNet_preprocessed\Dataset003_Hippocampus\dataset.json -pfile .\nnUNet_preprocessed\Dataset003_Hippocampus\nnUNetPlans.json .\nnUNet_preprocessed\Dataset003_Hippocampus\gt_segmentations\ .\nnUNet_results\Dataset002_Hippocampus\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all_pred_final\
```

about 87.4% dice accuracy in test set

## Optimization

1. dataset.json channel_names zscore specify to MRI to customize normalization https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md#how-to-implement-custom-normalization-strategies
2. also use region-based training with "regions_class_order" adding to dataset.json https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
3. hiddenlayer to generate neural network structure https://github.com/waleedka/hiddenlayer
4. set "nnUNet_n_proc_DA" training threads
5. change 5-fold cross validation split https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
6. 