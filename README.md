# Can We Find Neurons that Cause Unrealistic Images in Deep Generative Networks?
*** Our paper is accepted at IJCAI-2022 !! 

### Artifact Detection/Correction - Official PyTorch Implementation


![](https://github.com/hichoe95/Artifact-Detection-and-Sequential-Ablation/blob/main/figure/Screen%20Shot%202022-01-18%20at%202.48.49%20PM.png?raw=true)

This repo provides the official PyTorch implementation of the following paper:

> **Can We Find Neurons that Cause Unrealistic Images in Deep Generative Networks?**  
> [Hwanil Choi](https://github.com/hichoe95), [Wonjoon Chang](https://github.com/onejoon), [Jaesik Choi](http://sailab.kaist.ac.kr/members/jaesik/)*  
> Korea Advanced Institute of Science and Technology, **KAIST**  
>
  
> **Abstract**  
> Even though image generation with Generative Adversarial Networks (GANs) has been showing remarkable ability to generate high-quality images, GANs do not always guarantee photorealistic images will be generated. Sometimes they generate images that have defective or unnatural objects, which are referred to as 'artifacts'. Research to determine why the artifacts emerge and how they can be detected and removed has not been sufficiently carried out. To analyze this, we first hypothesize that rarely activated neurons and frequently activated neurons have different purposes and responsibilities for the progress of generating images. By analyzing the statistics and the roles for those neurons, we empirically show that rarely activated neurons are related to failed results of making diverse objects and lead to artifacts. In addition, we suggest a correction method, called 'sequential ablation', to repair the defective part of the generated images without complex computational cost and manual efforts.  
> https://arxiv.org/abs/2201.06346



## Dependencies
- PyTorch 1.4.0
- python 3.6
- cuda 10.0.x
- cudnn 7.6.3


## Pre-Trained Weights (Official) - [GenForce](https://github.com/genforce/genforce)
|Dataset \ Model| PGGAN | StyleGAN2 |
|:---:|:---:|:---:|
|CelebA-HQ (Official)| [1024 x 1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW_3jQ6E7xlKvCSHYrbmkQQBAB8tgIv5W5evdT6-GuXiWw?e=gRifVa&download=1)| X |
|FFHQ (Official)| X | [1024 X 1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EX0DNWiBvl5FuOQTF4oMPBYBNSalcxTK0AbLwBn9Y3vfgg?e=Q0sZit&download=1) |
|LSUN-Church (Official)| [256 x 256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQ8cKujs2TVGjCL_j6bsnk8BqD9REF2ME2lBnpbTPsqIvA?e=zH55fT&download=1)| [256 x 256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQzDtJUdQ4ROunMGn2sZouEBmNeFX4QWvxjermVE5cZvNA?e=tQ7r9r&download=1)|
|LSUN-CAT (Official)|[256 x 256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQdveyUNOMtAue52n6BxoHoB6Yup5-PTvBDmyfUn7Un4Hw?e=7acGbT&download=1) | [256 x 256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EUKXeBwUUbZJr6kup7PW4ekBx2-vmTp8FjcGb10v8bgJxQ?e=nkerMF&download=1) |

For following implementation, download **StyleGAN2 FFHQ** weights in current directory. Otherwise, you should change the '--weight_path' option to your directory.

*More pre-trained weights are available in [genforce-model-zoo](https://github.com/genforce/genforce/blob/master/MODEL_ZOO.md)*
- optional : [StyleGAN3](https://github.com/NVlabs/stylegan3)

## Implementation

- **Options**
```
optional arguments:
  -h, --help                show this help message and exit
  --gpu GPU                 gpu index number
  --batch_size BATCH_SIZE
                            batch size for pre processing and generating process
  --sample_size SAMPLE_SIZE
                            sample size for statistics
  --freq_path FREQ_PATH
                            loading saved frequencies of neurons
  --model MODEL             pggan, styelgan2
  --dataset DATASET         ffhq, cat, church, etc
  --resolution RESOLUTION
                            dataset resolution
  --weight_path WEIGHT_PATH
                            pre-trained weight path
  --detection DETECTION
                            implement normal/artifact detection
  --correction CORRECTION
                            implement correction task

```

- **Usage**
```
python main.py --gpu 0 --batch_size 30 --sample_size 30000 --freq_pth ./stats \
               --model stylegan2 --dataset ffhq --resolution 1024 --weight_path ./ \
               --detection True --correction True      
```

*__If you are on remote server, then to show the results, you should do the following. (X11 forwarding).__*
- [X11 forwarding](https://kb.iu.edu/d/bdnt)  


*__You can also implement our codes in 'Jupyter Notebook' that has more degree of freedom.
Use the 'notebook.ipynb' file.__*


## Detection results for 50K samples

### Bottom 60 images
![](https://github.com/hichoe95/Artifact-Detection-and-Sequential-Ablation/blob/main/figure/norm.png?raw=true)

### Top 60 images
![](https://github.com/hichoe95/Artifact-Detection-and-Sequential-Ablation/blob/main/figure/arti.png?raw=true)

## Correction results & Masking part of artifact
![](https://github.com/hichoe95/Artifact-Detection-and-Sequential-Ablation/blob/main/figure/masks.png?raw=true)
![](https://github.com/hichoe95/Artifact-Detection-and-Sequential-Ablation/blob/main/figure/correction.png?raw=true)

## Citation
```
@misc{choi2022neurons,
      title={Can We Find Neurons that Cause Unrealistic Images in Deep Generative Networks?}, 
      author={Hwanil Choi and Wonjoon Chang and Jaesik Choi},
      year={2022},
      eprint={2201.06346},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## LICENSE
MIT LICENSE
