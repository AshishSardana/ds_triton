#!/bin/bash

cd /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models

# download trafficcamnet
mkdir -p ../../models/tlt_pretrained_models/trafficcamnet && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/resnet18_trafficcamnet_pruned.etlt \
    -O ../../models/tlt_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned.etlt && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/trafficnet_int8.txt \
    -O ../../models/tlt_pretrained_models/trafficcamnet/trafficnet_int8.txt

# download vehicletypenet
mkdir -p ../../models/tlt_pretrained_models/vehicletypenet && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_vehicletypenet/versions/pruned_v1.0/files/resnet18_vehicletypenet_pruned.etlt \
    -O ../../models/tlt_pretrained_models/vehicletypenet/resnet18_vehicletypenet_pruned.etlt && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_vehicletypenet/versions/pruned_v1.0/files/vehicletypenet_int8.txt \
    -O ../../models/tlt_pretrained_models/vehicletypenet/vehicletypenet_int8.txt

# download dashcamnet
mkdir -p ../../models/tlt_pretrained_models/dashcamnet && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_dashcamnet/versions/pruned_v1.0/files/resnet18_dashcamnet_pruned.etlt \
    -O ../../models/tlt_pretrained_models/dashcamnet/resnet18_dashcamnet_pruned.etlt && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_dashcamnet/versions/pruned_v1.0/files/dashcamnet_int8.txt \
    -O ../../models/tlt_pretrained_models/dashcamnet/dashcamnet_int8.txt

# download vehiclemakenet
mkdir -p ../../models/tlt_pretrained_models/vehiclemakenet && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_vehiclemakenet/versions/pruned_v1.0/files/resnet18_vehiclemakenet_pruned.etlt \
    -O ../../models/tlt_pretrained_models/vehiclemakenet/resnet18_vehiclemakenet_pruned.etlt && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_vehiclemakenet/versions/pruned_v1.0/files/vehiclemakenet_int8.txt \
    -O ../../models/tlt_pretrained_models/vehiclemakenet/vehiclemakenet_int8.txt

#################################################################################################################################

cd ~/..

# move config files and run deepstream app for generating tensorrt engines
git clone https://github.com/AshishSardana/ds_triton.git

cp /ds_triton/engine_bs/config_infer_primary_trafficcamnet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/config_infer_primary_trafficcamnet.txt

cp /ds_triton/engine_bs/deepstream_app_source1_trafficcamnet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/deepstream_app_source1_trafficcamnet.txt

cp /ds_triton/engine_bs/config_infer_secondary_vehicletypenet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/config_infer_secondary_vehicletypenet.txt

cp /ds_triton/engine_bs/deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt

deepstream-app -c /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/deepstream_app_source1_trafficcamnet.txt

deepstream-app -c /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt

#################################################################################################################################

# prepare video samples
apt-get update && apt-get install -y ffmpeg

cd /opt/nvidia/deepstream/deepstream-5.1/samples/

./prepare_classification_test_video.sh

# move the engine, config.pbtxt and label files for trafficcamnet
cd /opt/nvidia/deepstream/deepstream-5.1/samples/

mkdir -p trtis_model_repo/trafficcamnet/1
cp models/tlt_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned.etlt_b50_gpu0_int8.engine trtis_model_repo/trafficcamnet/1/resnet18_trafficcamnet_pruned.etlt_b50_gpu0_int8.engine
cp /ds_triton/trafficcamnet_config.pbtxt /opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo/trafficcamnet/config.pbtxt
cp /ds_triton/labels_trafficcamnet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo/trafficcamnet/labels.txt

# move the engine, config.pbtxt and label files for vehicletypenet
cd /opt/nvidia/deepstream/deepstream-5.1/samples/

mkdir -p trtis_model_repo/Secondary_VehicleTypes/1
cp models/tlt_pretrained_models/vehicletypenet/resnet18_vehicletypenet_pruned.etlt_b200_gpu0_int8.engine trtis_model_repo/Secondary_VehicleTypes/1/resnet18_vehicletypenet_pruned.etlt_b200_gpu0_int8.engine
cp /ds_triton/vehicletypenet_config.pbtxt /opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo/Secondary_VehicleTypes/config.pbtxt
cp /ds_triton/labels_vehicletypenet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo/Secondary_VehicleTypes/labels.txt

# move the model and app config files of the use-case
cp /ds_triton/config/config_infer_primary_trafficcamnet_triton.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_primary_trafficcamnet_triton.txt
cp /ds_triton/config/config_infer_secondary_plan_engine_vehicletypes.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_secondary_plan_engine_vehicletypes.txt
cp /ds_triton/config/source1_primary_trafficcamnet_vehicletypenet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/source1_primary_trafficcamnet_vehicletypenet.txt

# launch application
deepstream-app -c /opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/source1_primary_trafficcamnet_vehicletypenet.txt
