# ProjectNeura: SegSTRONG-C

## Usage

### Build Docker Image

```shell
docker build --no-cache ./ -t segstrongc:projnura
```

### Create Docker Container

```shell
docker run --ipc=host --rm -v "F:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_nnunet:/workspace/data" --gpus="device=0" -it segstrongc:projnura
```

### Transfer Data

```shell
python data_transfer.py
```

### Set Environment Variables

```shell
export nnUNet_raw=/workspace/data/nnUNet_raw
export nnUNet_preprocessed=/workspace/data/nnUNet_preprocessed
export nnUNet_results=/workspace/data/nnUNet_weights
```

### Preprocess

```shell
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -pl nnUNetPlannerResEncM
```

### Train

```shell
nnUNetv2_train 1 2d 0 -p nnUNetResEncUNetMPlans -device cuda --npz
nnUNetv2_train 1 2d 1 -p nnUNetResEncUNetMPlans -device cuda --npz
nnUNetv2_train 1 2d 2 -p nnUNetResEncUNetMPlans -device cuda --npz
nnUNetv2_train 1 2d 3 -p nnUNetResEncUNetMPlans -device cuda --npz
nnUNetv2_train 1 2d 4 -p nnUNetResEncUNetMPlans -device cuda --npz
```
