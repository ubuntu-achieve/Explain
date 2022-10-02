import os
import cv2
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import numpy as np
from skimage.segmentation import mark_boundaries
from lime import lime_image
print('Notebook run using keras:', keras.__version__)


def transform_img_fn(img_path):
    x = image.load_img(img_path, target_size=(H, W))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

H, W = 224, 224 # 输入图片的尺寸
# 选择是否展示
is_show   = False
# 测试图片路径
root_path = 'Images'
# 保存图片路径
output_path = 'Results/BayLime'
if __name__ == "__main__":
    # 检查保存文件夹是否存在，若不存在则新建一个
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    inet_model = VGG19(include_top=True, weights='imagenet')
    for img_path in os.listdir(root_path):
        images = transform_img_fn(os.path.join(root_path, img_path))
        preds = inet_model.predict(images)
        for x in decode_predictions(preds)[0]:
            print(x)
        explainer = lime_image.LimeImageExplainer(feature_selection='none')#kernel_width=0.1

        explanation = explainer.explain_instance(images[0], inet_model.predict,
                                                top_labels=3, hide_color=0, batch_size=10,
                                                num_samples=200,model_regressor='non_Bay')
        #'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior','BayesianRidge_inf_prior_fit_alpha'

        #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=True)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        cv2.imwrite(os.path.join(output_path, 'baylime_vgg19_'+img_path), mark_boundaries(temp / 2 + 0.5, mask))

        #print(explanation.as_list(explanation.top_labels[0]))