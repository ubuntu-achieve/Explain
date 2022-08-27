%Install dependency library%
pip install -r ./requirements.txt
python .\CAM\all_cam.py
python .\CAM\cam.py
python .\IG\IGOS_main.py
python .\LRP\LRP_densenet_main.py
python .\LRP\LRP_resnet_main.py
python .\CAM\Relevance-CAM.py --models vgg19 --target_layer 52
python .\CAM\Relevance-CAM.py --models resnet50 --target_layer layer2