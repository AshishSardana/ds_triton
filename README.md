# Scaling DeepStream-Triton on NVIDIA A100 MIG instances through Kubernetes

This repository contains configuration files to run an end-to-end video analytics application on DeepStream v5.1. The application locates 4 different objects on the road (car, pedestrian, roadsign and bicycle) and then classifies the cars into 6 different classes - sedan, minivan, truck etc.
We are using TrafficCamNet for object detection and VehicleTypeNet for classification, both of which are pre-trained models available on NVIDIA GPU Cloud.
The deployment is first done on a single 2g.10gb MIG instance of a NVIDIA A100 GPU then scaled all the way up to 8 x A100's, all configured with the same MIG profile.

## Pre-requisite:
1. A server with 1 or more (preferably 8) NVIDIA A100's, either on cloud (AWS p4dn.24xlarge) or on-prem.
2. GPUs sliced with 2g.10gb MIG profile. 
2. NVIDIA Driver 460+
3. Docker image - nvcr.io/nvidia/deepstream:5.1-21.02-triton

## Instructions to run:
1. Launch the docker container
`docker run -it --rm --gpus device=<MIG-instance-UUID> nvcr.io/nvidia/deepstream:5.1-21.02-triton`

2. Clone the git repository
`git clone https://github.com/AshishSardana/ds_triton.git`

3. Execute the automate script
`cd ds_triton && bash automate_script.sh`

## Expected output:
On a 2g.10gb MIG instance, this application would run at 30 frames per second for 35 full HD video streams.
