import json
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval
from PIL import Image
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import find_contours
from tqdm import tqdm, trange
import time
from tool.JoyTool import *

'''
maskrcnn_resnet50_fpn模型
'''
is_show_plot = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res_json_path = 'output/segmentation_maskrcnn_results.json'

data_dir = "D:/0_Joy_Data/1_Work/Projects/PyCharmProjects/1_dataset/COCO2017/val2017"
ann_file = "D:/0_Joy_Data/1_Work/Projects/PyCharmProjects/1_dataset/COCO2017/annotations/instances_val2017.json"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)

def coco_ins_seg(imgIds):
    model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    results = []
    joy_timer = JoyTimer()
    for img_id in tqdm(imgIds):
        image_info = coco.loadImgs(img_id)[0]
        image_path = f'{data_dir}/{image_info["file_name"]}'
        data_size_list.append(os.path.getsize(image_path) / 1024)
        image_tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            prediction = model(image_tensor)[0]
        masks = prediction['masks'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        if is_show_plot:
            image = Image.open(image_path).convert("RGB")
            fig, ax = plt.subplots(1, figsize=(12, 9))
            ax.imshow(image)
            plt.axis('off')
        for mask, score, box, label in zip(masks, scores, boxes, labels):
            mask = mask[0]
            mask_binary = mask > 0.5
            rle_mask = encode(np.array(mask_binary, order="F", dtype=np.uint8))
            rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
            result = {
                "image_id": img_id,
                "category_id": label,
                "segmentation": rle_mask,
                "score": float(score)
            }
            results.append(result)
            threshold = 0.5
            if is_show_plot and score > threshold:
                contour = find_contours(mask_binary, 0.5)[0]
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
                x, y, w, h = box
                rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
            plt.show()
        joy_timer.record()
    time_res = joy_timer.show()
    with open(res_json_path, 'w') as f:
        json.dump(results, f, cls=CustomJSONEncoder)
    return time_res


def coco_eval(coco):
    coco_pred = coco.loadRes(res_json_path)
    coco_eval = COCOeval(coco, coco_pred, 'segm')
    for img_id in imgIds:
        coco_eval.params.imgIds = [img_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        accuracy_list.append(coco_eval.stats[0])


if __name__ == '__main__':
    data_size_list = []  # 存储图像大小
    accuracy_list = []  # 存储精度
    coco = COCO(ann_file)
    imgIds = sorted(coco.getImgIds())
    imgIds.insert(0, 139)  # 在正式测试前插入一个图像用于预热程序，避免第一次装载模型等造成的冷启动时间
    imgIds=imgIds[1:3]
    infer_time_list = coco_ins_seg(imgIds)  # 执行所有任务的推理，保存推理结果，返回推理时间
    coco_eval(coco)  # 在推理完所有图像后统一计算精度

    del imgIds[0]  # 删除第一张预热样本的数据
    del data_size_list[0]
    del infer_time_list[0]
    del accuracy_list[0]

    file_output = [imgIds, data_size_list, infer_time_list, accuracy_list]
    write_multi_list_to_txt(file_output, ['id', 'data_size', 'infer_time', 'accuracy'],filename='maskrcnn')
