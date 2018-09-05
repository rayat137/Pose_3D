# Pose_3D
Exploiting temporal information for 3D pose estimation

This code is released for the paper "Exploiting temporal information for "Exploiting temporal information for 3D human
pose estimation", accepted for ECCV 2018. https://arxiv.org/pdf/1711.08585.pdf


### Dependencies

* [h5py](http://www.h5py.org/)
* [tensorflow](https://www.tensorflow.org/) 1.0 or later



### Training from the scratch

To train from the scratch use the command:

`python temporal_3d.py --use_sh --camera_frame --dropout 0.5`

### Pre-trained model

You can download a pre-trained model for testing, visualization and fine-tuning from: 
https://drive.google.com/file/d/1j2jpwDpfj5NNx8n1DVqCIAESNTDZ2BDf/view?usp=sharing

Download and untar the file. Copy the contents in `Pose_3D/temporal_3d_release/trained_model/All/dropout_0.5/epochs_100/adam/lr_1e-05/linear_size1024/batch_size_32/use_stacked_hourglass/seqlen_5/`

### Evaluate the model

To evaluate the pre-trained model call: 

`python temporal_3d.py --use_sh --camera_frame --dropout 0.5 --load 1798202 --evaluate`

Pass the global iteration number (in this case the model is for iteration 1798202) to the load flag to load the model of corresponding iteration.


(Documentation to be finished soon)



