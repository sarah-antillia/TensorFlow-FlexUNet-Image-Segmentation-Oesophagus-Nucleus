<h2>TensorFlow-FlexUNet-Image-Segmentation-Oesophagus-Nucleus (2025/08/22)</h2>

This is the first experiment of Image Segmentation for Oesophagus, 
 based on our 
 <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
<b>TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass)</b></a>
, and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1MtmOTJUBfLplgT88wLFHvBXZf8n7dVph/view?usp=sharing">
<b>Augmented-Oesophagus-PNG-ImageMask-Dataset.zip</b></a>.
which was derived by us from 
<br><br>
<a href="https://www.kaggle.com/datasets/ipateam/nuinsseg">
NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H&E-Stained Images.
</a>
<br>
<br>

On the derivation of the augmented dataset, please refer to our experiment 
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-Oesophagus">
Tensorflow-Image-Segmentation-Pre-Augmented-Oesophagus
</a>.
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a> ,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<b>Acutual Image Segmentation for 512x512 pixels Oesophagus images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/10.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/17.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/barrdistorted_1002_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/barrdistorted_1002_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/barrdistorted_1002_0.3_0.3_8.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been take from the following kaggle web site 
<a href="https://www.kaggle.com/datasets/ipateam/nuinsseg/data">
NuInsSeg
</a><br><br>

<b>About Dataset</b><br>
NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H&E-Stained Histological Images<br>
<br>
<b>Citation</b><br>
@article{mahbod2023nuinsseg,<br>
  title={NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H\&E-Stained Histological Images},<br>
  author={Mahbod, Amirreza and Polak, Christine and Feldmann, Katharina and Khan, Rumsha and Gelles, <br>
  Katharina and Dorffner, Georg and Woitek, Ramona and Hatamikia, Sepideh and Ellinger, Isabella},<br>
  journal={arXiv preprint arXiv:2308.01760},<br>
  year={2023}<br>
}
<br>
<br>
<b>Content</b><br>
The NuInsSeg dataset contains more than 30k manually segmented nuclei from 31 
human and mouse organs and 665 image patches extracted from H&E-stained whole slide images. 
We also provide ambiguous area masks for the entire dataset to show in which areas manual 
semantic/instance segmentation were impossible.
<br><br>
<b>Human organs:</b><br>
cerebellum, cerebrum (brain), colon (rectum), epiglottis, jejunum, kidney, liver, lung, melanoma, 
muscle, oesophagus, palatine tonsil, pancreas, peritoneum, placenta, salivary gland, 
spleen, stomach (cardia), stomach (pylorus), testis, tongue, umbilical cord, and urinary bladder
<br><br>
<b>Mouse organs:</b><br>
cerebellum, cerebrum, colon, epiglottis, lung, melanoma, muscle, peritoneum, stomach (cardia), 
stomach (pylorus), testis, umbilical cord, and urinary bladder)
<br>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">
Attribution 4.0 International (CC BY 4.0)
</a>
<br>
<br>
<h3>
<a id="2">
2 Oesophagus ImageMask Dataset
</a>
</h3>
 If you would like to train this Oesophagus Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1MtmOTJUBfLplgT88wLFHvBXZf8n7dVph/view?usp=sharing">
Augmented-Oesophagus-PNG-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Oesophagus
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Oesophagus Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Oesophagus/Oesophagus_Statistics.png" width="512" height="auto"><br>
<br>

As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Oesophagus TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Oesophagus/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Oesophagus and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Oesophagus 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; RGB colors     Oesophagus-nucleus;white     
rgb_map = {(0,0,0):0,(255,255,255):1,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py) </a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 26,27,28)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 53,54,55)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 55.<br><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/train_console_output_at_epoch55.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Oesophagus/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Oesophagus/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Oesophagus</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Oesophagus.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/evaluate_console_output_at_epoch55.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Oesophagus/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Oesophagus/test was low and dice_coef_multiclass 
high as shown below.
<br>
<pre>
categorical_crossentropy,0.0941
dice_coef_multiclass,0.9542
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Oesophagus</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Oesophagus.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Oesophagus/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/17.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/18.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/18.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/18.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/32.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/32.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/32.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/barrdistorted_1004_0.3_0.3_17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/barrdistorted_1004_0.3_0.3_17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/barrdistorted_1004_0.3_0.3_17.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/deformed_alpha_1300_sigmoid_8_41.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/deformed_alpha_1300_sigmoid_8_41.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/deformed_alpha_1300_sigmoid_8_41.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/images/distorted_0.02_rsigma0.5_sigma40_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oesophagus/mini_test_output/distorted_0.02_rsigma0.5_sigma40_6.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Improving generalization capability of deep learning-based nuclei instance segmentation <br>
by non-deterministic train time and deterministic test time stain normalization
</b><br>
Amirreza Mahbod, Georg Dorffner, Isabella Ellinger, Ramona Woitek, Sepideh Hatamikia<br>

<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10825317/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC10825317/
</a>
<br>
<br>
<b>2. Tensorflow-Image-Segmentation-Pre-Augmented-Oesophagus</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-Oesophagus">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-Oesophagus
</a>
<br>
<br>

