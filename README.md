# SMPL AMC/ASF Imitator

**NOTE**: the code style is awful and I don't have time to prettify it. If
anyone is really going to use this, please open an issue and I'll have a release
version when I have time.

For a given AMC/ASF motion sequence, we transfer the motion to SMPL model, and generate a corresponding 3D SMPL sequence.

This work is based on [my implmentation](https://github.com/CalciferZh/SMPL) of [SMPL model](http://smpl.is.tue.mpg.de/) and [my implementation](https://github.com/CalciferZh/AMCParser) of AMC/ASF parser.

## Demo

### Skeleton (left: SMPL target, right: ASF/AMC source)
![Skeleton Demo](./demo_skeleton.gif)

### Skinned Model
![Skinned Demo](./demo_skinned.gif)

## Usage
Run `python 3Dviewer.py` to see demo.

Also, run `python batch.py` to extract all poses into `./pose/` from `./data/`.

## Challenge
The skeleton of SMPL is a little bit different from CMU MoCap Dataset's. In this implementation, we only process femur and tibia and ignore other differences. We first pose SMPL skeleton (specifically legs) to be in the same pose with ASF defination. After that, we extract rotation matrices from AMC files and apply them to the aligned SMPL model.

Feel free to [contact me](mailto:calciferzh@outlook.com) for more details.
