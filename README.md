# Official RCS-YOLO
This is the source code for the paper, "RCS-YOLO: A Fast and High-Accuracy Object Detector for Brain Tumor Detection", of which I am the first author.

## Model
The model configuration (i.e., network construction) file is [rcs-yolo.yaml] in the the directory ./cfg/training/.

#### Training

The hyperparameter setting file is [hyp_training.yaml] in the [hyp_training.yaml].

######Single GPU training
```
python train.py --workers 8 --device 0 --batch-size 32 --data data/mydata.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

######Multiple GPU training
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/br35h.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

#### Testing

The model weights we pretrained on the brain tumor detection was saved as [best.pt] in the directory ./rcs-yolo/runs/train/exp/weights/.
```
python test.py --data data/br35h.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/exp/weights/best.pt --name val
```

## Performance
We trained and evaluated RCS-YOLO on the dataset [Br35H :: Brain Tumor Detection 2020](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection). We used 500 images of which in the ’train’ folder were selected as the training set, while the other 201 images in the ’val’ folder as the testing set. The best results are shown in bold.
| Model | Params | Precision | Recall | AP<sub>50</sub> | AP<sub>50:95</sub> | FLOPs | FPS |
| :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| [YOLOv6-L](https://github.com/meituan/YOLOv6) | 59.6M | 0.907 | 0.920 | 0.929 | 0.709 | 150.5G | 64.0 |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | 36.9M | 0.897 | **0.955** | 0.929 | 0.709 | 150.5G | 71.4 |
| [YOLOv8l](https://github.com/ultralytics/ultralytics) | 43.9M | 0.907 | 0.919 | 0.944 | **0.731** | 164.8G | 76.2 |
| **RCS-YOLO** | 45.7M | **0.936** | 0.945 | **0.946** | 0.729 | **94.5G** | **114.8** |

The screenshot/visualisation of evaluation results are in the directory ./runs/val/.

<!---
## Suggested Citation>
Please cite our paper if you use code from this repository:
> Plain Text

*Nature* Style
```
Kang, M., Ting, C.-M., Ting, F. F., and Phan, R. Rcs-yolo: A fast and high-accuracy object detector for brain tumor detection. In , (2023)
```

*IEEE* Style
```
M. Kang, C.-M. Ting, F. F. Ting, and R. Phan, "Rcs-yolo: A fast and high-accuracy object detector for brain tumor detection," in , 2023.
```

> BibTeX Format
```
@inproceedings{kang2023rcsyolo,
  author = "Ming Kang and Chee-Ming Ting and Fung Fung Ting and Raphael Phan",
  title = "RCS-YOLO: A Fast and High-Accuracy Object Detector for Brain Tumor Detection",
  booktitle = " ",
  year = "2023"
}
```
```
@inproceedings{kang2023rcsyolo,
  author = {Ming Kang and Chee-Ming Ting and Fung Fung Ting and Raphael Phan},
  title = {RCS-YOLO: A Fast and High-Accuracy Object Detector for Brain Tumor Detection},
  booktitle = { },
  year = {2023}
}
```

## License
RCS-YOLO is released under the Apache 2.0 license. Please see the [LICENSE](https://github.com/mkang315/rcs-yolo/blob/main/LICENSE) file for more information.
-->
## References
Many utility codes of our project references the codes of [YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv5](https://github.com/ultralytics/yolov5) repos.
