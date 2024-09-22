# Official RCS-YOLO
This is the source code for the paper titled "RCS-YOLO: A Fast and High-Accuracy Object Detector for Brain Tumor Detection" published in the [Proceedings](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_57) of the 26th International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI 2023](https://conferences.miccai.org/2023/en)), of which I am the first author. The paper is available to download from [Springer](https://link.springer.com/content/pdf/10.1007/978-3-031-43901-8_57?pdf=chapter%20toc) or [arXiv](https://arxiv.org/pdf/2307.16412).

## Model
The RepVGG/RepConv ShuffleNet You Only Look Once (RCS-YOLO) model configuration (i.e., network construction) file is rcs-yolo.yaml (2 heads) or rcs3-yolo.yaml (3 heads) in the directory [./cfg/training/](https://github.com/mkang315/RCS-YOLO/tree/main/cfg/training). The [RepVGG](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)/[RepConv](https://arxiv.org/pdf/2207.02696.pdf) [ShuffleNet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf) based [One-Shot Aggregation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_CenterMask_Real-Time_Anchor-Free_Instance_Segmentation_CVPR_2020_paper.pdf) (RCS-OSA) module file is rcsosa.py in the directory [./models/](https://github.com/mkang315/RCS-YOLO/tree/main/models), which is the unique module we proposed.

#### Installation
Install requirements.txt with recommended dependencies Python >= 3.8 environment including Torch <= 1.7.1 and CUDA <= 11.1:
```
pip install -r requirements.txt
```

#### Training

The hyperparameter setting file is hyp_training.yaml in the directory [./data/](https://github.com/mkang315/RCS-YOLO/tree/main/data).

###### Single GPU training
```
python train.py --workers 8 --device 0 --batch-size 32 --data data/br35h.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

###### Multiple GPU training
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/br35h.yaml --img 640 640 --cfg cfg/training/rcs-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp_training.yaml
```

#### Testing

The model weights we pretrained on the brain tumor detection was saved as best.pt in the directory [./runs/train/exp/weights/](https://github.com/mkang315/RCS-YOLO/tree/main/runs/train).
```
python test.py --data data/br35h.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/exp/weights/best.pt --name val
```

## Evaluation
We trained and evaluated RCS-YOLO on the dataset [Br35H :: Brain Tumor Detection 2020](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection). The .txt format annotations in the folder dataset-Br35H are coverted from original json format. We used 500 images of which in the ’traindata’ folder were selected as the training set, while the other 201 images in the ’valdata’ folder as the testing set. <br />

The screenshots of evaluation results were saved in the directory [./runs/val/](https://github.com/mkang315/RCS-YOLO/tree/main/runs/val).

## Ablation Study

The files train_repvgg-csp.py and https://test_repvgg-csp.py are used for an ablation study of the comparison network repvgg-csp.yaml in the directory [./cfg/ablation/](https://github.com/mkang315/RCS-YOLO/tree/main/cfg/ablation).

## Referencing Guide
Please cite our paper if you use code from this repository. Here is a guide to referencing this work in various styles for formatting your references:
> Plain Text

- Springer Reference Style</br>
Kang, M., Ting, C.-M., Ting, F.F., Phan, R.C.-W.: RCS-YOLO: a fast and high-accuracy object detector for brain tumor detection. In: Greenspan, H., et al. (eds.) MICCAI 2023. LNCS, vol. 14223, 600–610. Springer, Cham (2023). [https://doi.org/10.1007/978-3-031-43901-8_57](https://doi.org/10.1007/978-3-031-43901-8_57)</br>
<sup>**NOTE:** MICCAI conference proceedings are part of the book series LNCS in which Springer's format for bibliographical references is strictly enforced. This is important, for instance, when citing previous MICCAI proceedings. LNCS stands for Lecture Notes in Computer Science.</sup>

- Nature Referencing Style</br>
Kang, M., Ting, C.-M., Ting, F. F. & Phan, R. C.-W. RCS-YOLO: a fast and high-accuracy object detector for brain tumor detection. In *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2023: 26th International Conference, Vancouver, Canada, October 8–12, 2023, Proceedings, Part IV* (eds. Greenspan, H. et al.) 600–610 (Springer, 2023).</br>

- IEEE Reference Style</br>
M. Kang, C.-M. Ting, F. F. Ting, and R. C.-W. Phan, "Rcs-yolo: A fast and high-accuracy object detector for brain tumor detection," in *Proc. Int. Conf. Med. Image Comput. Comput. Assist. Interv. (MICCAI)*, Vancouver, BC, Canada, Oct. 8–12, 2023, pp. 600–610.</br>
<sup>**NOTE:** City of Conf., Abbrev. State, Country, Month & Day(s) are optional.</sup>

- Elsevier Numbered Style</br>
M. Kang, C.-M. Ting, F.F. Ting, R.C.-W. Phan, RCS-YOLO: A fast and high-accuracy object detector for brain tumor detection, in: H. Greenspan, A. Madabhushi, P. Mousavi, S. Salcudean, J. Duncan, T. Syeda-Mahmood, et al. (Eds.), Proceedings of the 26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 8–12 October 2023, Vancouver, BC, Canada, Springer, Cham, Germany, 2023, pp. 600–610.</br>
<sup>**NOTE:** Day(s) Month Year, City, Abbrev. State, Country of Conference, Publiser, and Place of Publication are optional.</sup>

- Harvard (Name–Date) Style</br>
Kang, M., Ting, C.-M., Ting, F.F. & Phan, R.C.-W., 2023. RCS-YOLO: A fast and high-accuracy object detector for brain tumor detection. In: H. Greenspan, A. Madabhushi, P. Mousavi, S. Salcudean, J. Duncan, T. Syeda-Mahmood, et al. (Eds.), Proceedings of the 26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 8–12 October 2023, Vancouver, BC, Canada. Springer, Cham, Germany, pp. 600–610.</br>
<sup>**NOTE:** Day(s) Month Year, City, Abbrev. State, Country of Conference, Publiser, and Place of Publication are optional.</sup>

> BibTeX Format</br>
```
\begin{thebibliography}{1}
\bibitem{Kang23Rcsyolo} Kang, M., Ting, C.-M., Ting, F.F., Phan, R.C.-W.: RCS-YOLO: a fast and high-accuracy object detector for brain tumor detection. In: Greenspan, H., et al. (eds.) MICCAI 2023. LNCS, vol. 14223, 600--610. Springer, Cham (2023). {\UrlFont https://doi.org/10.1007/978-3-031-43901-8\_57}
\end{thebibliography}
```
```
@inproceedings{Kang23Rcsyolo,
  author = "Kang, Ming and Ting, Chee-Ming and Ting, Fung Fung and Phan, Rapha{\"e}l C.-W.",
  title = "RCS-YOLO: a fast and high-accuracy object detector for brain tumor detection",
  editor = "Greenspan, Hayit and et al.",
  booktitle = "Medical Image Computing and Computer-Assisted Intervention – MICCAI 2023: 26th International Conference, Vancouver, Canada, October 8--12, 2023, Proceedings, Part IV",
  series = "Lecture Notes in Computer Science (LNCS)",
  volume = "14223",
  pages = "600--610",
  publisher = "Springer",
  address = "Cham",
  year = "2023",
  doi= "10.1007/978-3-031-43901-8\_57",
  url = "https://doi.org/10.1007/978-3-031-43901-8_57"
}
```
```
@inproceedings{Kang23Rcsyolo,
  author = "Ming Kang and Chee-Ming Ting and Fung Fung Ting and Rapha{\"e}l C.-W. Phan",
  title = "Rcs-yolo: A fast and high-accuracy object detector for brain tumor detection",
  booktitle = "Proc. Int. Conf. Med. Image Comput. Comput. Assist. Interv. (MICCAI)",
  address = "Vancouver, BC, Canada, Oct. 8--12",
  pages = "600--610",
  year = "2023",
}
```
<sup>**NOTE:** Please remove some optional *BibTeX* fields, for example, `series`, `volume`, `address`, `url` and so on, while the *LaTeX* compiler produces an error. Author names may be manually modified if not automatically abbreviated by the compiler under the control of the .bst file if applicable which defines bibliography/reference style. `Kang23Rcsyolo` could be `b1`, `bib1`, or `ref1` when references appear in numbered style in which they are cited. The quotation mark pair `""` in the field could be replaced by the brace `{}`. </sup>

## License
RCS-YOLO is released under the GNU General Public License v3.0. Please see the [LICENSE](https://github.com/mkang315/RCS-YOLO/blob/main/LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [YOLOv7](https://github.com/WongKinYiu/yolov7), [RepVGG](https://github.com/DingXiaoH/RepVGG), [ShuffleNet](https://github.com/megvii-model/ShuffleNet-Series), and [VoVNetV2](https://github.com/youngwanLEE/vovnet-detectron2) repositories.
