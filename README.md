# ProjectNeura: SegSTRONG-C

## Usage

### Build Docker Image

```shell
docker build ./ -t segstrongc:projnura
```

### Create Docker Container

```shell
docker run --rm -v "F:/SharedDatasets/SegSTRONGC_release:/workspace/data" --gpus="device=0" -it segstrongc:projnura
```

### Transfer Data

```shell
python data_transfer.py
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
