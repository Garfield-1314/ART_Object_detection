import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import numpy as np 
import os 
from utils import yolo_cfg
from model_resnet import yolo_eval
import cv2
import argparse
from PIL import Image,ImageDraw
from evaluate import get_yolo_boxes

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 6)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-model', help='trained tflite', default=r'./yolo3_iou_smartcar_final.tflite', type=str)
    parser.add_argument('-input_dir', help='folder containing input images', default=r'../github/Mcx_Art_test_datasets/JPEGImages' ,type=str)
    parser.add_argument('-output_dir', help='output folder for results', default='test', type=str)
    args, unknown = parser.parse_known_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    labels = ['person']
    cfg = yolo_cfg()
    anchors_path = cfg.cluster_anchor
    anchors = get_anchors(anchors_path)
    anchors_num = len(anchors) / 3

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for f in os.listdir(args.input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(args.input_dir, f))

    # 遍历处理每张图片
    for image_path in image_files:
        try:
            image = Image.open(image_path)
            origin_image = image.copy()
            image_w = image.size[0]
            image_h = image.size[1]
            
            # 图像预处理
            scale = min(cfg.width/image_w, cfg.height/image_h)
            nw = int(image_w*scale)
            nh = int(image_h*scale)
            dx = (cfg.width-nw)//2
            dy = (cfg.height-nh)//2
            
            # 调整大小并填充
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (cfg.width, cfg.height), (128,128,128))
            new_image.paste(image, (dx, dy))
            
            # 准备输入数据
            image_data = np.array(new_image)/255.0
            image_data = np.expand_dims(image_data, axis=0)

            # 进行预测
            pred_boxes, scores, classes = get_yolo_boxes(args.model, anchors, image_data, 
                                                        (image_w, image_h), cfg.nms_score_threshold,cfg.nms_iou_threshold)

            # 绘制检测框
            draw = ImageDraw.Draw(origin_image)
            for box in pred_boxes:
                color = tuple(np.random.randint(0, 255, 3).tolist())
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)

            # 保存结果
            output_name = os.path.basename(image_path)
            output_path = os.path.join(args.output_dir, output_name)
            origin_image.save(output_path)
            print(f'Saved result to {output_path}')
            
        except Exception as e:
            print(f'Error processing {image_path}: {str(e)}')

    print('All images processed!')