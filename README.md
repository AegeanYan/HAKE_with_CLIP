## CPaStaNet: A CLIP-based Human Activity Knowledge Engine**

------

This repo contains code for the ICIPMC 2023 paper:

**CPaStaNet: A CLIP-based Human Activity Knowledge Engine**

*Yijia Hong, Haotian Luo, Xialin He, and Yaoqi Ye* ICIPMC 2023

------

A simple but efficient way to improve a general human activity feature extractor and human PaSta (part states) detector based on HAKE data.

------

#### Pre-requisities

We used python==3.7.2 and you can find more information in requirement.txt

#### Datasets

Please refer to [DATASET.md](https://github.com/AegeanYan/HAKE-Action-Torch-with-CLIP/blob/AegeanYan/HAKE-Action-Torch-with-CLIP/DATASET.md)

#### Train the model

To train the CPaStaNet model run the following command:

```
python main.py --eval=False
```

#### Evaluate

```
python main.py --eval=True
```

#### Citation

T.B.C