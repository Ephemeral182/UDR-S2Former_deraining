# <h2 style="margin-bottom:7px;margin-top:20px;font-weight:400;">Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks (ICCV'23)</h2> 

<a href="https://ephemeral182.github.io"><strong>Sixiang Chen</strong></a><sup>1,3</sup>*&nbsp;&nbsp;&nbsp; 
<a href="https://owen718.github.io">Tian Ye</a><sup>1,3</sup>&nbsp;&nbsp;&nbsp;
<a href="https://noyii.github.io">Jinbin Bai</a><sup>2</sup>&nbsp;&nbsp;&nbsp;
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=hWo1RTsAAAAJ">Erkang Chen</a><sup>3</sup>&nbsp;&nbsp;&nbsp;
Jun Shi<sup>4</sup>&nbsp;&nbsp;&nbsp;
<a href="https://sites.google.com/site/indexlzhu/home">Lei Zhu</a><sup>1,5 ✉️</sup>&nbsp;&nbsp;&nbsp;
<br>
<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)&nbsp;&nbsp;&nbsp;
<sup>2</sup>National University of Singapore&nbsp;&nbsp;&nbsp;<br>
<sup>3</sup>School of Ocean Information Engineering, Jimei University&nbsp;&nbsp;&nbsp;             
<sup>4</sup>Xinjiang University&nbsp;&nbsp;&nbsp; 
<sup>5</sup>The Hong Kong University of Science and Technology&nbsp;&nbsp;&nbsp;   

---
    
<div class="post-line" style=“border-top: 0.4rem solid #55657d;
    display: block;
    margin: 0 auto 2rem;
    width: 5rem;”></div>
    
<div style="margin-bottom: 0.7em;" class="post-authors">
                <div class="col-md-8 col-md-offset-2 text-center">
                    <ul class="nav nav-pills nav-justified" style="box-shadow:0 0">
                        <li>
                            <a href="https://ephemeral182.github.io/UDR_S2Former_deraining/">
                            <!-- <a href="https://arxiv.org/abs/2112.05504"> -->
                            <img class="post-logo" src="https://ephemeral182.github.io/images/paper.jpg" height="50px">
                                <h5><strong>arXiv</strong></h5>
                            </a>
                        </li>
                        <li>
                             <a href="https://ephemeral182.github.io/UDR_S2Former_deraining/">
                            <img class="post-logo" src="https://ephemeral182.github.io/images/paper.jpg" height="50px">
                                <h5><strong>ICCV 2023</strong></h5>
                            </a>
                        </li>
                        <li>
                             <a href="https://ephemeral182.github.io/UDR_S2Former_deraining/">
                            <img class="post-logo" src="https://ephemeral182.github.io/images/datatset.jpg" height="50px">
                                <h5><strong>Dataset</strong></h5>
                            </a>
                        </li>                        
                        <li>
                            <a href="https://ephemeral182.github.io/UDR_S2Former_deraining/">
                            <img class="post-logo" src="https://ephemeral182.github.io/images/github.png" height="50px">
                                <h5><strong>Code</strong></h5>
                            </a>
                        </li>
                        <li>
                             <a href="https://ephemeral182.github.io/UDR_S2Former_deraining/">
                            <img class="post-logo" src="https://ephemeral182.github.io/images/supplementary.jpg" height="50px">
                                <h5><strong>Supplementery</strong></h5>
                            </a>
                        </li>
                    </ul>
                </div>
        </div>

## <h2 class="post-section" style="padding-left: 25px;    margin-bottom: 0px;    padding-top: 20px;">Abstract</h2>

 
In the real world, image degradations caused by rain often exhibit a combination of rain streaks and raindrops, thereby increasing the challenges of recovering the underlying clean image. Note that the rain streaks and raindrops have diverse shapes, sizes, and locations in the captured image, and thus modeling the correlation relationship between irregular degradations caused by rain artifacts is a necessary prerequisite for image deraining. 
This paper aims to present an efficient and flexible mechanism to learn and model degradation relationships in a global view, thereby achieving a unified removal of intricate rain scenes. 
To do so, we propose a <u>S</u>parse <u>S</u>ampling Trans<u>former</u> based on <u>U</u>ncertainty-<u>D</u>riven <u>R</u>anking, dubbed <strong>UDR-S<sup>2</sup>Former</strong>. 
Compared to previous methods, our UDR-S<sup>2</sup>Former has three merits. First, it can adaptively sample relevant image degradation information to model underlying degradation relationships. 
Second, explicit application of the uncertainty-driven ranking strategy can facilitate the network to attend to degradation features and understand the reconstruction process. 
Finally, experimental results show that our UDR-S<sup>2</sup>Former clearly outperforms state-of-the-art methods for all benchmarks.


<h2 class="post-section" style="
    padding-left: 25px;
    margin-bottom: 0px;
    padding-top: 20px;
">Method</h2>

<img  style="width:80;max-width:100%" src="https://ephemeral182.github.io/images/uncertainty_map.png" alt="Left Image">
<img  style="width:40;max-width:50%" src="https://ephemeral182.github.io/images/udr_overview1.png" alt="Left Image">
<img  style="width:40;max-width:50%" src="https://ephemeral182.github.io/images/udr_overview2.png" alt="Right Image">



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
