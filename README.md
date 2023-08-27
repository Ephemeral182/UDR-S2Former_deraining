# Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks (ICCV'23)


<a href="https://ephemeral182.github.io"><strong>Sixiang Chen*</strong></a>&nbsp;&nbsp;&nbsp; 
<a href="https://owen718.github.io">Tian Ye</a>*&nbsp;&nbsp;&nbsp;
<a href="https://noyii.github.io">Jinbin Bai</a>&nbsp;&nbsp;&nbsp;
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=hWo1RTsAAAAJ">Erkang Chen</a>&nbsp;&nbsp;&nbsp;
Jun Shi&nbsp;&nbsp;&nbsp;
<a href="https://sites.google.com/site/indexlzhu/home">Lei Zhu</a><sup>✉️</sup>&nbsp;&nbsp;&nbsp;
<br>

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
[![supplement](https://img.shields.io/badge/Supplementary-Material-B85252)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
[![project](https://img.shields.io/badge/Project-Presentation-F9D371)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
<!--[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1-L43wj-VTppkrR9AL6cPBJI2RJi3Hc_z/view?usp=sharing)-->


<table style="margin: auto;">
  <tr>
    <td> <img src = "https://github.com/Ephemeral182/UDR-S2Former_deraining/blob/main/image/real_gif1.gif" width="388"> </td>
    <td> <img src = "https://github.com/Ephemeral182/UDR-S2Former_deraining/blob/main/image/real_gif2.gif" width="388"> </td>
  </tr>
</table>

<table style="margin: auto;">
  <tr>
    <td style="text-align: center;">
      <img 
        src="https://github.com/Ephemeral182/UDR-S2Former_deraining/blob/main/image/350.gif" 
        width="804" 
      >
    </td>
  </tr>
</table>

---
    
<div class="post-line" style=“border-top: 0.4rem solid #55657d;
    display: block;
    margin: 0 auto 2rem;
    width: 5rem;”></div>
    

## Abstract

> *In the real world, image degradations caused by rain often exhibit a combination of rain streaks and raindrops, thereby increasing the challenges of recovering the underlying clean image. Note that the rain streaks and raindrops have diverse shapes, sizes, and locations in the captured image, and thus modeling the correlation relationship between irregular degradations caused by rain artifacts is a necessary prerequisite for image deraining. 
This paper aims to present an efficient and flexible mechanism to learn and model degradation relationships in a global view, thereby achieving a unified removal of intricate rain scenes. 
To do so, we propose a <u>S</u>parse <u>S</u>ampling Trans<u>former</u> based on <u>U</u>ncertainty-<u>D</u>riven <u>R</u>anking, dubbed <strong>UDR-S<sup>2</sup>Former</strong>. 
Compared to previous methods, our UDR-S<sup>2</sup>Former has three merits. First, it can adaptively sample relevant image degradation information to model underlying degradation relationships. 
Second, explicit application of the uncertainty-driven ranking strategy can facilitate the network to attend to degradation features and understand the reconstruction process. 
Finally, experimental results show that our UDR-S<sup>2</sup>Former clearly outperforms state-of-the-art methods for all benchmarks.*


## <h2  style="padding-left: 25px;    margin-bottom: 0px;    padding-top: 20px;">Method</h2>

<img  style="width:100%;max-width:96%" src="https://ephemeral182.github.io/images/uncertainty_map.png" alt="Left Image">
<div class="post-img-group">
    <img class="post-img" style="width:48%;max-width:50%" src="https://ephemeral182.github.io/images/udr_overview1.png" alt="Left Image">
    <img class="post-img" style="width:48%;max-width:50%" src="https://ephemeral182.github.io/images/udr_overview2.png" alt="Right Image">
  </div> 
  </div>



<h2 class="post-section" style="
    text-align: center;
    padding-left: 25px;
    margin-bottom: 10px;
    padding-top: 20px;
">Quantitative Comparison</h2>
<div style="box-shadow:3px 6px 13px 0px  rgba(0,0,0,0.5)">
  <div class="post-img-group">
    <img class="post-img" style="max-width:100%;margin-bottom:0;" src="https://ephemeral182.github.io/images/metric.png"  alt="Left Image">
  </div>
  </div>

<table style="margin: auto;">
  <tr>
    <td> <img src = "https://github.com/Ephemeral182/UDR-S2Former_deraining/blob/main/image/real4.gif" height="250"> </td>
    <td> <img src = "https://github.com/Ephemeral182/UDR-S2Former_deraining/blob/main/image/real9.gif" height="250"> </td>
  </tr>
</table>

## Installation
Our UDR-S<sup>2</sup>Former is built in Pytorch2.0.1, we train and test it on Ubuntu20.04 environment (Python3.8+, Cuda11.6).

:satisfied: For installing, please follow these instructions:
```
conda create -n py38 python=3.8
conda activate py38
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt  
```
## Dataset
We train and test our UDR-S<sup>2</sup>Former in Rain200H<strong>(Rain streaks)</strong>, Rain200L<strong>(Rain streaks)</strong>, RainDrop<strong>(Raindrops&Rain streaks)</strong> and AGAN<strong>(Raindrops)</strong> benchmarks. The download links of datasets are provided.

<table>
  <tr>
    <th align="left">Dataset</th>
    <th align="center">Rain200H</th>
    <th align="center">Rain200L</th>
    <th align="center">RainDrop</th>
    <th align="center">AGAN</th>
  </tr>
  <tr>
    <td align="left">Link</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Download</a></td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Download</a></td>
    <td align="center"><a href="https://github.com/Songforrr/RainDS_CCN">Download</a></td>
    <td align="center"><a href="https://github.com/rui1996/DeRaindrop">Download</a></td>
  </tr>
 </table>

## Quick Run

:raised_hands: To test the demo of [Deraining](https://github.com/Ephemeral182/UDR-S2Former_deraining/blob/main/pretrained/udrs2former_demo.pth)  on your own images simply, 
run:
```
python demo.py -c config/demo.yaml
```
:point_right: Here is an example to perform demo, please save your rainy images into the path of **‘image_demo/input_images’**, then execute the following command:
```
python demo.py -c config/demo.yaml
```
Then deraining results will be output to the save path of **'image_demo/output_images'**.

## Training Stage

:yum: Our training process is built upon pytorch_lightning, rather than the conventional torch framework. Please run the code below to begin training UDR-S<sup>2</sup>Former on various benchmarks (raindrop_syn,raindrop_real,agan,rain200h,rain200l). Example usage to training our model in raindrop_real:
```python
python train.py fit -c config/config_pretrain_raindrop_real.yaml
```
The logs and checkpoints are saved in ‘**tb_logs/udrs2former**‘.

## Inference Stage

:smile: We have pre-trained models available for evaluating on different datasets. Please run the code below to obtain the performance on various benchmarks via --dataset_type (raindrop_syn,raindrop_real,agan,rain200h,rain200l). Here is an example to test raindrop_real datatset:
```python
python3  test.py --dataset_type raindrop_real --dataset_raindrop_real your path
```
The rusults are saved in ‘**out/dataset_type**‘.


## Citation 
```
@inproceedings{chen2023deraining,
  title={Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks},
  author={Chen, Sixiang and Ye, Tian and Bai, Jinbin and Chen, Erkang and Shi, Jun and Zhu, Lei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
## Contact
If you have any questions, please contact the email sixiangchen@hkust-gz.edu.cn or ephemeral182@gmail.com.
