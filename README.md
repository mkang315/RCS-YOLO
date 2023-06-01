# Official RCS-YOLO
This is the source code for the paper, "RCS-YOLO: A Fast and High-Accuracy Object Detector for Brain Tumor Detection", of which I am the first author.

## Model
The model configuration (i.e., network construction) file is rcs-yolo.yaml in the the directory ./cfg/training/. The [RepVGG](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)/[RepConv](https://arxiv.org/pdf/2207.02696.pdf) [ShuffleNet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf) based [One-Shot Aggregation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_CenterMask_Real-Time_Anchor-Free_Instance_Segmentation_CVPR_2020_paper.pdf) (RCS-OSA) module file is rcsosa.py in the directory ./models/, which is the unique module we proposed.

#### Training

The hyperparameter setting file is hyp_training.yaml in the directory ./data/.

###### Single GPU training
```
python train.py --workers 8 --device 0 --batch-size 32 --data data/br35h.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

###### Multiple GPU training
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/br35h.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

#### Testing

The model weights we pretrained on the brain tumor detection was saved as best.pt in the directory ./rcs-yolo/runs/train/exp/weights/.
```
python test.py --data data/br35h.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/exp/weights/best.pt --name val
```

## Performance
We trained and evaluated RCS-YOLO on the dataset [Br35H :: Brain Tumor Detection 2020](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection). The txt format annotations in the folder dataset-Br35H are coverted from original json format. We used 500 images of which in the ’train’ folder were selected as the training set, while the other 201 images in the ’val’ folder as the testing set. The best results are shown in bold.<br />
<br />
**Table 1 Quantitative results of different methods. The best results are shown in bold.** 
| Model | Params | Precision | Recall | AP<sub>50</sub> | AP<sub>50:95</sub> | FLOPs | FPS |
| :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| [YOLOv6-L](https://github.com/meituan/YOLOv6) | 59.6M | 0.907 | 0.920 | 0.929 | 0.709 | 150.5G | 64.0 |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | 36.9M | 0.897 | **0.955** | 0.944 | 0.725 | 103.3G | 71.4 |
| [YOLOv8l](https://github.com/ultralytics/ultralytics) | 43.9M | 0.907 | 0.919 | 0.944 | **0.731** | 164.8G | 76.2 |
| **RCS-YOLO** | 45.7M | **0.936** | 0.945 | **0.946** | 0.729 | **94.5G** | **114.8** |

The screenshot/visualisation of evaluation results are in the directory ./runs/val/.

## Ablation Studies
The results of ablation studies are shown below in Markdown format. We demonstrate the effectiveness of the proposed RCS-OSA module in YOLO-based object detector.<br />
<br />
**Table 2 Ablation study on proposed RCS-OSA module.** 
| Method | Precision | Recall | AP<sub>50</sub> | AP<sub>50:95</sub> |
| :--------: | :-------: | :-------: | :-------: | :-------: |
| YOLOv4-CSP (w/o RCS-OSA) | 0.920 | 0.915 | 0.937 | 0.690 |
| YOLOv4-CSP (w/ RCS-OSA) | 0.927  | 0.919 | 0.939 | 0.703 |

## License
RCS-YOLO is released under the Apache 2.0 license. Please see the LICENSE file for more information.

## Acknowledgement
Many utility codes of our project references the codes of [YOLOv7](https://github.com/WongKinYiu/yolov7), [RepVGG](https://github.com/DingXiaoH/RepVGG), [ShuffleNet](https://github.com/megvii-model/ShuffleNet-Series) and [VoVNetV2](https://github.com/youngwanLEE/vovnet-detectron2) repositories.
