# Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks (ICCV'23)

<a href="https://ephemeral182.github.io"><strong>Sixiang Chen</strong></a><sup></sup>&nbsp;&nbsp;&nbsp; 
<a href="https://owen718.github.io">Tian Ye</a><sup></sup>&nbsp;&nbsp;&nbsp;
<a href="https://noyii.github.io">Jinbin Bai</a><sup></sup>&nbsp;&nbsp;&nbsp;
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=hWo1RTsAAAAJ">Erkang Chen</a><sup>3</sup>&nbsp;&nbsp;&nbsp;
Jun Shi<sup>4</sup>&nbsp;&nbsp;&nbsp;
<a href="https://sites.google.com/site/indexlzhu/home">Lei Zhu</a><sup>✉️</sup>&nbsp;&nbsp;&nbsp;

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
[![supplement](https://img.shields.io/badge/Supplementary-Material-B85252)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
[![project](https://img.shields.io/badge/Project-Presentation-F9D371)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
<!--[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1-L43wj-VTppkrR9AL6cPBJI2RJi3Hc_z/view?usp=sharing)-->

<div class="post-img-group">
    <img class="post-img" style="left:0;right:0;margin-bottom:0px;max-width:50%" src="https://github.com/Ephemeral182/UDR-S2Former_deraining/tree/main/image/real_gif1.gif" alt="Left Image">
    <img class="post-img" style="left:0;right:0;margin-bottom:0px;max-width:50%" src="https://github.com/Ephemeral182/UDR-S2Former_deraining/tree/main/image/real_gif2.gif" alt="Right Image">
    <img class="post-img" style="left:0;right:0;margin-bottom:0;width:100%" src="https://github.com/Ephemeral182/UDR-S2Former_deraining/tree/main/image/350.gif" alt="Left Image">
</div> 

  
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

## Installation
Our SnowFormer is built in Pytorch1.12.0, we train and test it ion Ubuntu20.04 environment (Python3.8, Cuda11.6).

For installing, please follow these intructions.
```
conda create -n py38 python=3.8
conda activate py38
conda install pytorch=1.12 
pip install opencv-python tqdm tensorboardX ....
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

To test the pre-trained models of [Deblurring](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view?usp=sharing), [Deraining](https://drive.google.com/file/d/1O3WEJbcat7eTY6doXWeorAbQ1l_WmMnM/view?usp=sharing), [Denoising](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view?usp=sharing) on your own images, run 
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```
Here is an example to perform Deblurring:
```
python demo.py --task Deblurring --input_dir ./samples/input/ --result_dir ./samples/output/
```
 
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
