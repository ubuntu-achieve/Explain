%Install dependency library%
pip install -r ./requirements.txt
python ./CAM/all_cam.py
python ./CAM/cam.py
python ./IG/IGOS_main.py
python ./LRP/LRP_densenet_main.py
python ./LRP/LRP_resnet_main.py
python ./CAM/Relevance-CAM.py --models vgg19 --target_layer 52
python ./CAM/Relevance-CAM.py --models resnet50 --target_layer layer2
python ./SmoothGraad/SmoothGrad_resnet.py
python ./SmoothGraad/SmoothGrad_vgg.py
python ./GNL/main.py -md guided_integrated_gradients -mn resnet50 -si 1 -ie 1 -ei 5 -gn 0 -da ./Images/
python ./GNL/main.py -md guided_integrated_gradients -mn vgg19 -si 1 -ie 1 -ei 5 -gn 0 -da ./Images/