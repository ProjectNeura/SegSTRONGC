# ProjectNeura: SegSTRONG-C

## Usage

### Build Docker Image

```shell
docker build --no-cache ./ -t segstrongc:projnura
```

### Create Docker Container

```shell
docker run --ipc=host --rm -v "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_nnunet:/workspace/data" --gpus="device=0" -it segstrongc:projnura
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
nnUNetv2_train 1 2d all -p nnUNetResEncUNetMPlans -device cuda
```

### Predict

```shell
nnUNetv2_predict -i /workspace/val/smoke_nnunet -o /workspace/val/smoke_predicted -d 1 -c 2d -p nnUNetResEncUNetMPlans -f all --save_probabilities
nnUNetv2_predict -i /workspace/val/lb_nnunet -o /workspace/val/lb_predicted -d 1 -c 2d -p nnUNetResEncUNetMPlans -f all --save_probabilities
nnUNetv2_predict -i /workspace/val/blood_nnunet -o /workspace/val/blood_predicted -d 1 -c 2d -p nnUNetResEncUNetMPlans -f all --save_probabilities
```
