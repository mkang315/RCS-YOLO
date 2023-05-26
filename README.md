# Official RCS-YOLO
PyTorch implementation of "RCS-YOLO: A Fast and High-Accuracy Object Detector for Brain Tumor Detection".

## Model
The architecture configuration of the model is [rcs-yolo.yaml](https://github.com/mkang315/rcs-yolo/blob/main/yaml/training/rsc-yolo.yaml) in the the directory ./yaml/training/. The model weights we pretrained on the brain tumor detection is [best.pt](https://github.com/mkang315/rcs-yolo/blob/main/runs/train/exp/weights/best.pt) in the directory ./rcs-yolo/runs/train/exp/weights/.

> Training
> Testing
```
python test.py --data data/mydata.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/exp/weights/best.pt --name rcs_yolo_640_val
```

## Performance
We trained and evaluated RCS-YOLO on the dataset [Br35H :: Brain Tumor Detection 2020](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).

## Suggested Citation
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

## License
RCS-YOLO is released under the Apache 2.0 license. Please see the [LICENSE](https://github.com/mkang315/rcs-yolo/blob/main/LICENSE) file for more information.
