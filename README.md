# 2D-Hand-Pose-Estimation-Pytorch

## Description
This project aims to develop an artificial intelligence system that can estimate hand pose from 2D images. The system utilizes deep learning techniques to accurately detect and predict the position and orientation of a hand in an image.

## Requirements
Python 3.8.1 (recommended)

Virtual environment (recommended)

## Installation
1. Create a virtual environment:
```bash
python -m venv myenv
```
2. Activate the virtual environment:
```bash
source myenv/bin/activate  # for Unix-based systems
```
```bash
myenv\Scripts\activate    # for Windows
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Usage

To train or test the network, you have two options:

Use the Jupyter notebook [TFG_Training.ipynb](TFG_Training.ipynb) for training or testing the hand pose estimation model.

Alternatively, you can run the [main.py](main.py) script to perform training or testing directly from the command line.

## Contact
For any questions or inquiries, please contact [josperrod9](https://github.com/josperrod9).

## References
List any references, papers, or resources related to the project.

[1] Santavas N, Kansizoglou I, Bampis L, et al. Attention! a lightweight 2d hand pose estimation approach[J]. IEEE Sensors Journal, 2020. [[code]][https://github.com/nsantavas/Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach]

[2] Chen Y, Ma H, Kong D, et al. Nonparametric structure regularization machine for 2D hand pose estimation[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2020: 381-390. [code][https://github.com/HowieMa/NSRMhand] (This project is manually forked from this project. )

[3] Wei S E, Ramakrishna V, Kanade T, et al. Convolutional pose machines[C]//Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2016: 4724-4732.

[4] Simon T, Joo H, Matthews I, et al. Hand keypoint detection in single images using multiview bootstrapping[C]//Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2017: 1145-1153. [Panoptic][http://domedb.perception.cs.cmu.edu/handdb.html]

[5] Zimmermann C, Ceylan D, Yang J, et al. Freihand: A dataset for markerless capture of hand pose and shape from single rgb images[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 813-822. [FreiHAND][https://lmb.informatik.uni-freiburg.de/projects/freihand/]

[6] Zhang J, Jiao J, Chen M, et al. 3d hand pose tracking and estimation using stereo matching[J]. arXiv preprint arXiv:1610.07214, 2016. [SHP]

[7] Shivakumar S H, Oberweger M, Rad M, et al. HO-3D: A Multi-User, Multi-Object Dataset for Joint 3D Hand-Object Pose Estimation[J]. arXiv. org e-Print archive, 2019. [HO3D_v2]

