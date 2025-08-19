# SMOTracker：A SAM2-based Multiple Small-object Tracker

- (2025.08) We provide an initial version of our work.

## Abstract
Existing multi-object tracking (MOT) methods typically follow a tracking-by-detection paradigm with Kalman filters. Detected bounding boxes are fed into the Kalman filter’s observation equation to update tracklet locations, and similarity metrics are used to associate detections with tracklets. However, for small objects with low visual quality, detectors often fail to produce reliable results in consecutive frames or only yield low-confidence detections. Missing detections interrupt the observation update, and when the motion trajectories of small objects change, their tracks are prone to being lost. Camera zooming or movement further exacerbates this issue. To address these challenges, we propose SMOTracker, a SAM2-based multi-object tracker designed for small objects. Specifically, we replace the Kalman filter with the Segment Anything Model 2 (SAM2) and adopt a masklet-to-detection association strategy. Predicting object positions via SAM2 alleviates the limitations of Kalman filters, while assigning identities (IDs) to masklets instead of detections mitigates track loss under detection failures. Since SAM2 tracks objects sequentially, completing one object before starting another, and cannot be directly combined with detection-based approaches, we segment the video into multiple clips while maintaining a shared SAM2 memory bank. We further introduce cross-clip ID assignment by associating SAM2 masklets with detector outputs, enabling SMOTracker to operate effectively in an online streaming setting. Extensive experiments on multiple small-object tracking benchmarks demonstrate that SMOTracker achieves robust performance across diverse tracking scenarios.

## Model Overview
<p align="center">
<img src="assets/structure.jpg" width="500"/>
</p>

## Tracking Performance
The comparison video

https://github.com/user-attachments/assets/daa1b7f0-ec60-4b62-90f7-bd727dc657eb

TOPICTrack video

https://github.com/user-attachments/assets/c8896041-69b4-40f7-9621-69b5eef0ddc0

G-SMOTracker video

https://github.com/user-attachments/assets/c832a017-3f2b-4fea-9203-3c0622e1a220

Y-SMOTracker video

https://github.com/user-attachments/assets/145433d5-7de0-4e37-97ac-1da309933971



### Evaluation Results
We use [BoxMOT](https://github.com/mikel-brostrom/boxmot) as comparison platform and take 3 datasets for evaluation. This repository includes 8 multiple object tracking methods. Their detector is with the same yolov10-x checkpoint which can be downloaded in the Installation section . Our Y-SMOTracker also uses this yolov10-x checkpoint.  
#### Results of dataset DOHA Anti-UAV
| Method           | HOTA          | IDF1          | IDs       | IDSW      | IDFN        |
|------------------|---------------|---------------|-----------|-----------|-------------|
| DeepOCSORT       | 38.553        | 35.799        | 487       | 486       | 17262       |
| BoTSORT          | 39.386        | 39.326        | 847       | 827       | 17268       |
| StrongSORT       | 33.295        | 40.15         | 811       | 832       | 17268       |
| OCSORT           | 35.843        | 46.3          | 128       | 120       | 17262       |
| ByteTrack        | 38.223        | 49.156        | 147       | 139       | 17220       |
| Imprassoc        | 33.412        | 48.158        | 1459      | 1620      | 17250       |
| Boosttrack       | 11.215        | 7.629         | 67        | 48        | 22680       |
| Boosttrack++     | 13.127        | 8.569         | 75        | 54        | 22026       |
| OPICTrack        | 23.453        | 32.927        | **30**    | 45        | 7204        |
| **G-SMOTracker** | <u>61.109</u> | <u>65.685</u> | 48        | **36**    | <u>6243</u> |
| **Y-SMOTracker** | **75.109**    | **77.591**    | <u>32</u> | <u>40</u> | **3957**    |

#### Results of dataset Anti-UAV
| Method           | HOTA          | IDF1          | IDs        | IDSW      | IDFN         |
|------------------|---------------|---------------|------------|-----------|--------------|
| DeepOCSORT       | 55.481        | <u>54.934</u> | 579        | 577       | 23229        |
| BoTSORT          | 54.981        | 52.957        | 652        | 656       | 23521        |
| StrongSORT       | 43.359        | 38.838        | 809        | 1004      | 30486        |
| OCSORT           | 52.060        | 49.696        | 713        | 729       | 26002        |
| ByteTrack        | <u>55.575</u> | 53.842        | 700        | 683       | 23138        |
| Imprassoc        | 39.971        | 31.895        | 1137       | 1181      | 33187        |
| Boosttrack       | 24.092        | 19.705        | 450        | 426       | 42202        |
| Boosttrack++     | 24.176        | 19.803        | 452        | 425       | 42157        |
| TOPICTrack       | 27.216        | 21.136        | 449        | 414       | 41655        |
| **G-SMOTracker** | 43.286        | 53.582        | <u>150</u> | **21**    | <u>13038</u> |
| **Y-SMOTracker** | **61.578**    | **81.571**    | **94**     | <u>33</u> | **7981**     |

#### Results of dataset DUT Anti-UAV
| Method           | HOTA          | IDF1          | IDs       | IDSW      | IDFN        |
|------------------|---------------|---------------|-----------|-----------|-------------|
| DeepOCSORT       | <u>42.401</u> | <u>46.262</u> | 62        | 47        | 8171        |
| BoTSORT          | 40.822        | 41.737        | 60        | 45        | 8506        |
| StrongSORT       | 39.379        | 41.241        | 87        | 76        | 8686        |
| OCSORT           | 41.078        | 44.894        | 82        | 68        | 8437        |
| ByteTrack        | 39.934        | 41.128        | 76        | 63        | 8563        |
| Imprassoc        | 32.791        | 30.892        | 268       | 243       | 9720        |
| Boosttrack       | 12.698        | 9.4363        | **31**    | 21        | 12428       |
| Boosttrack++     | 13.75         | 10.473        | 34        | 24        | 12345       |
| TOPICTrack       | 16.904        | 18.642        | 38        | 24        | 11673       |
| **G-SMOTracker** | 36.566        | 39.531        | 33        | **8**     | <u>7557</u> |
| **Y-SMOTracker** | **53.098**    | **55.931**    | <u>32</u> | <u>16</u> | **5890**    |

## Installation

1. Run `python path_initialize.py` 
2. Install env by `pip install -r requirements.txt`
3. Install `ffmpeg` into system referring to the [download webpage](https://ffmpeg.org/download.html).
4. Install the environments of `SAM2` as individual project outside this project, and `Grounding-DINO` under the `/SMOTracker`.
   * For `Grounding-DINO` installation, after clone, if there is a problem of importing `"_C"`, you can get into `GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu`, go to line `65` and `135`, modify `value.type()` to `value.scalar_type()`, back to the path `GroundingDINO`, run `python setup.py build install`, and wait for the successful compiling of `"_C"`.
5. Download
[YOLOV10-x checkpoint](https://drive.google.com/file/d/134OtEnjhvGCF06FPIHzIyElAAHSZEkPM/view?usp=drive_link)
trained by us, 
[sam2.1_hiera_large checkpoints](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
, and
[Grounding-DINO checkpoint](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
to the `weights` directory.
6. Download [bee24_AGW.pth](https://drive.google.com/file/d/1OJRRERPh0uOv8sbEhDIofGFScigT1Y-w/view?usp=sharing) from [TOPICTrack](https://github.com/holmescao/TOPICTrack) to `external/weights`.



## Run the demo
You can download the
[Test video](https://drive.google.com/file/d/1TOussiXyNZ6JY7xVqgI9s3r5TJS_NPev/view?usp=drive_link)
and place it at e.g., `buffer/video`.
* To run G-SMOTracker, run `python G_SMOTracker_run.py`
* To run Y-SMOTracker, run `python Y_SMOTracker_run.py`
* The output video is at `buffer/video_outpu`


## Discussion
https://github.com/user-attachments/assets/a336d874-db6d-4ad2-8173-bcb6b81a97b5

The video is the result of the Grounded SAM2 model. As shown in the video and described in the related Internet, Grounded SAM2 can only track the objects from the first frame and is unable to add new objects that appear in subsequent frames. Furthermore, Grounded SAM2 is not efficient for tracking small objects such as UAVs.  Our original test video contains 1,159 frames. If we use Grounded SAM2 to process the entire video, it cannot track any objects due to its inability to detect objects in the first frame. Therefore, we take out the last clip of the video. Then we use Grounded SAM2 to process it. From the video results shown above, it can be seen that since there are only four tracked objects in the first frame, only these four objects are tracked in the video.
SAMRUAI, like Grounded SAM2, can only track objects annotated in the first frame.

In contrast, both G-SMOTracker and Y-SMOTracker are capable of tracking UAVs in the all frames and add new IDs if there are new objects in later frames. Therefore, Grounded SAM2 and SAMRUAI are not directly comparable to our methods in terms of performance metrics of multiple object tracking. 

## Related work
[SAM2](https://github.com/facebookresearch/sam2): The Segment Anything Model (SAM) facilitates object tracking in video via segmentation. Notably, it lacks the capability to incorporate and track newly emerging objects during processing.

[SAMURAI](https://github.com/yangchris11/samurai): An enhanced variant of SAM2 significantly improves object tracking robustness following occlusion events. Notably, this model retains two key limitations of SAM2: the inability to track multiple objects concurrently and the incapability to track newly appearing objects.

[Grounded SAM2](https://github.com/IDEA-Research/Grounded-SAM-2): This project integrates SAM2 with Grounding-DINO to achieve video object tracking through a hybrid approach. However, the system exhibits three primary limitations: (1) Object detection is confined exclusively to the initial frame via Grounding-DINO; (2) Subsequent frames lack dynamic object detection capabilities, preventing identification of newly appearing entities; (3) Tracking performance proves notably inefficient for small objects due to inherent segmentation constraints.

## Other test videos
Here we present more video result from other videos in our dataset.

Multiple objects in sky

https://github.com/user-attachments/assets/8d5e9823-264a-4d03-aa9f-d490cb7390f0

Urban scenario

https://github.com/user-attachments/assets/4a77fb33-10da-41d6-9e79-23367e6531bd

Sea scenario

https://github.com/user-attachments/assets/fdbfd5c8-b07e-4873-ba3c-ed506c70a47d

Multiple objects above sea

https://github.com/user-attachments/assets/04661d6b-7624-4c13-be1b-77dc0a1f1cbf