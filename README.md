## CPaStaNet: A CLIP-based Human Activity Knowledge Engine**

------

This repo contains code for the ICIPMC 2023 paper:

**CPaStaNet: A CLIP-based Human Activity Knowledge Engine**

*Yijia Hong, Haotian Luo, Xialin He, and Yaoqi Ye* ICIPMC 2023

------

A simple but efficient way to improve a general human activity feature extractor and human PaSta (part states) detector based on HAKE data.

We employ a new framework called CLIP to replace the CNNs extracting features layers before Activity2Vev in order to extract information from the picture more efficiently. We also propose several PaSta-based activity reasoning methods. The whole framework is known as CPaStaNet. Promoted by it, our method achieves significant advancements, e.g., 17.9 and 24.4 mAP in image-level human activity and part state classification tasks on the HAKE-Image dataset. 

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

### Our result

![](https://raw.githubusercontent.com/AegeanYan/ImageBed/main/%E6%88%AA%E5%B1%8F2023-02-07%20%E4%B8%8B%E5%8D%8811.58.48.png)


#### Citation

T.B.C