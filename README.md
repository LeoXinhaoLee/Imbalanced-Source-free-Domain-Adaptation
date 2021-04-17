# ISFDA-master
The codes will be released soon.

### Prerequisites

------

- python == 3.7.10

- pytorch == 1.2.0

- torchvision == 0.4.0

- numpy, scipy, slearn, argparse, tqdm, PIL

### Dataset:

------

Please download the VisDA-C dataset, Office-Home dataset and DomainNet dataset from the following links:

- VisDA-C: http://csr.bu.edu/ftp/visda17/clf/

- Office-Home: https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view
- DomainNet: http://ai.bu.edu/M3SDA/

Organize the datasets into the following file structure where `ISFDA` is the parent folder of the datasets.

```
ISFDA
├── data
    ├── domainnet
    ├── office-home-RSUT
    ├── VisDA-RSUT-100
├── VisDA-RSUT-50
    └── VisDA-RSUT-10
```

We have put the txt files that record image paths and labels of each datasets we use in our paper in the corresponding folder under the data folder. By default, we assume the paths of DomainNet, Office-Home, VisDA-C are /data/DomainNet, /data/Office_home, and /data/VisDA_C respectively, but they could be costomized. 

Txt files that record the data split for VisDA-RSUT-100, VisDA-RSUT-50, VisDA-10 are created by us following the protocol of the [BBN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf​) paper for long-tail visual recognition.

I thank Shuhan Tan for providing the data split txt files for Office-Home (RSUT) and DomainNet in their [COAL](https://arxiv.org/abs/1910.10320) paper.

### Training

------

1. #####  Training on the Office-Home (RSUT) dataset

   - Train model on the source domain **Clipart**(s = 1)

    ```python
    cd object/
    python image_source.py --trte val --da uda --output ckps/source/ --gpu_id 0 --dset office-home-RSUT --max_epoch 20 --s 1 --source_balanced
    ```

   - Adaptation to other target domains **Product and Real World**, respectively

    ```python
    python image_target.py --cls_par 0.3 --da uda --output_src ckps/source/ --output ckps/target/ --gpu_id 0 --dset office --s 1 --max_epoch 15 --topk_ent --scd_label --intra_dense --inter_sep
    ```

2. ##### Training on the VisDA-C (RSUT) dataset

   - Synthetic-to-real 

    ```python
    cd object/
    python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-RSUT --net resnet101 --lr 1e-3 --max_epoch 10 --s 0 --source_balanced
    python image_target.py --cls_par 0.3 --da uda --dset VISDA-RSUT --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3 --max_epoch 15 --topk_ent --scd_label --intra_dense --inter_sep
    ```


3. **Training on the DomainNet dataset**

   - Train model on the source domain **Clipart**

   ```
   cd object/
    python image_source.py --trte val --da uda --output ckps/source/ --gpu_id 0 --dset domainnet --max_epoch 20 --s 0 --source_balanced
   ```

   - Adaptation to other target domains **Painting, Real , Sketch**, respectively

   ```python
    cd object/
    python image_target.py --cls_par 0.3 --da uda --dset domainnet --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3 --max_epoch 15 --topk --scd_label --intra_dense --inter_sep
   ```



