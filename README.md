# SMOTracker：A SAM2-based Multiple Small-object Tracker

- (2025.08) We provide an initial version of our work.

## Abstract
Existing multi-object tracking (MOT) methods typically follow a tracking-by-detection paradigm with Kalman filters. Detected bounding boxes are fed into the Kalman filter’s observation equation to update tracklet locations, and similarity metrics are used to associate detections with tracklets. However, for small objects with low visual quality, detectors often fail to produce reliable results in consecutive frames or only yield low-confidence detections. Missing detections interrupt the observation update, and when the motion trajectories of small objects change, their tracks are prone to being lost. Camera zooming or movement further exacerbates this issue. To address these challenges, we propose SMOTracker, a SAM2-based multi-object tracker designed for small objects. Specifically, we replace the Kalman filter with the Segment Anything Model 2 (SAM2) and adopt a masklet-to-detection association strategy. Predicting object positions via SAM2 alleviates the limitations of Kalman filters, while assigning identities (IDs) to masklets instead of detections mitigates track loss under detection failures. Since SAM2 tracks objects sequentially, completing one object before starting another, and cannot be directly combined with detection-based approaches, we segment the video into multiple clips while maintaining a shared SAM2 memory bank. We further introduce cross-clip ID assignment by associating SAM2 masklets with detector outputs, enabling SMOTracker to operate effectively in an online streaming setting. Extensive experiments on multiple small-object tracking benchmarks demonstrate that SMOTracker achieves robust performance across diverse tracking scenarios.

## Tracking performance

### Results on 3 benchmark test datasets

| Method       | HOTA   | IDF1   | IDs  | IDSW | IDFN  |
|--------------|--------|--------|------|------|-------|
| DeepOCSORT   | 38.553 | 35.799 | 487  | 486  | 17262 |
| BoTSORT      | 39.386 | 39.326 | 847  | 827  | 17268 |
| StrongSORT   | 33.295 | 40.15  | 811  | 832  | 17268 |
| OCSORT       | 35.843 | 46.3   | 128  | 120  | 17262 |
| ByteTrack    | 38.223 | 49.156 | 147  | 139  | 17220 |
| Imprassoc    | 33.412 | 48.158 | 1459 | 1620 | 17250 |
| Boosttrack   | 11.215 | 7.629  | 67   | 48   | 22680 |
| Boosttrack++ | 13.127 | 8.569  | 75   | 54   | 22026 |
| G-SMOTracker | 61.109 | 65.685 | 48   | 36   | 6243  |
| Y-SMOTracker | 75.109 | 77.591 | 32   | 40   | 3957  |
### Model structure
<p align="center"><img src="ReadMeFigs/structure.jpg" width="500"/></p>



| Method       | HOTA   | IDF1   | IDs  | IDSW | IDFN  |
|--------------|--------|--------|------|------|-------|
| DeepOCSORT   | 55.481 | 54.934 | 579  | 577  | 23229 |
| BoTSORT      | 54.981 | 52.957 | 652  | 656  | 23521 |
| StrongSORT   | 43.359 | 38.838 | 809  | 1004 | 30486 |
| OCSORT       | 52.060 | 49.696 | 713  | 729  | 26002 |
| ByteTrack    | 55.575 | 53.842 | 700  | 683  | 23138 |
| Imprassoc    | 39.971 | 31.895 | 1137 | 1181 | 33187 |
| Boosttrack   | 24.092 | 19.705 | 450  | 426  | 42202 |
| Boosttrack++ | 24.176 | 19.803 | 452  | 425  | 42157 |
| G-SMOTracker | 43.286 | 53.582 | 150  | 21   | 13038 |
| Y-SMOTracker | 61.578 | 81.571 | 84   | 33   | 7981  |

| Method       | HOTA   | IDF1   | IDs  | IDSW | IDFN  |
|--------------|--------|--------|------|------|-------|
| DeepOCSORT   | 42.401 | 54.934 | 579  | 577  | 23229 |
| BoTSORT      | 40.822 | 52.957 | 652  | 656  | 23521 |
| StrongSORT   | 39.379 | 38.838 | 809  | 1004 | 30486 |
| OCSORT       | 41.078 | 49.696 | 713  | 729  | 26002 |
| ByteTrack    | 39.934 | 53.842 | 700  | 683  | 23138 |
| Imprassoc    | 32.791 | 31.895 | 1137 | 1181 | 33187 |
| Boosttrack   | 12.698 | 19.705 | 450  | 426  | 42202 |
| Boosttrack++ | 13.75  | 19.803 | 452  | 425  | 42157 |
| G-SMOTracker | 36.566 | 53.582 | 150  | 21   | 13038 |
| Y-SMOTracker | 61.578 | 81.571 | 84   | 33   | 7981  |



