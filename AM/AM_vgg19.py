from keras import activations
from keras.applications import VGG19
from vis.utils import utils
from vis.visualization import visualize_saliency

import os
from PIL import Image
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 选择是否展示
is_show   = False
# 测试图片路径
root_path = 'Images'
# 保存图片路径
output_path = 'Results/AM'
# 实例化VGG19模型
model = VGG19(weights='imagenet', include_top=True)
# 取出模型中某层的索引
layer_idx = utils.find_layer_idx(model, 'predictions')

# 交换Softmax和线性层
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

def path_join(*path):
    join_path = path[0]
    for index in range(1, len(path)):
        join_path = os.path.join(join_path, path[index])
    return join_path.replace('\\', '/')
if __name__ == "__main__":
    # 检查保存文件夹是否存在
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    for i, img in enumerate(os.listdir(root_path)):
        img = utils.load_img(path_join(root_path, img),target_size=(224, 224))
        for modifier in ['guided', 'relu']:
            plt.suptitle(modifier)
            # 20是对应于ouzel的模型索引
            grads = visualize_saliency(
                model,
                layer_idx,
                filter_indices=20,
                seed_input=img,
                backprop_modifier=modifier
            )
            grads = Image.fromarray(grads)
            if is_show:
                grads.show()
            grads.save(path_join(output_path, "%s_vgg19_%d.JPEG"%(modifier, i+1)))