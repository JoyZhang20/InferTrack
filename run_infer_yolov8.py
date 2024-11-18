import json
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from tool.JoyTool import *
from contextlib import contextmanager
import sys


# 定义上下文管理器，用于重定向标准输出
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


'''
maskrcnn_resnet50_fpn模型
'''
is_show_plot = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res_json_path = 'output/'

data_dir = "D:/0_Joy_Data/1_Work/Projects/PyCharmProjects/1_dataset/COCO2017/val2017"  # 替换成自己的数据集路径
ann_file = "D:/0_Joy_Data/1_Work/Projects/PyCharmProjects/1_dataset/COCO2017/annotations/instances_val2017.json"


# 自定义JSON编码器,用于将np类型转换为Python的float类型保存到txt中
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)


def coco_ins_seg(imgIds, model_size):
    data_size_list = []  # 存储图像大小
    infer_time_list = []
    results = []
    model = YOLO('model/yolov8' + model_size + '.pt')
    for img_id in tqdm(imgIds):
        img_info = coco.loadImgs(img_id)[0]
        img_path = f'{data_dir}/{img_info["file_name"]}'
        data_size_list.append(os.path.getsize(img_path) / 1024)
        # YOLOv8进行推理
        results_img = model(img_path, verbose=False)
        speed_time = results_img[0].speed
        infer_time_list.append(round(speed_time['preprocess'] + speed_time['inference'] + speed_time['postprocess'], 4))
        preds = results_img[0].boxes

        for box in preds:
            x, y, w, h = box.xywh[0].tolist()
            xmin, ymin, width, height = x - w / 2, y - h / 2, w, h
            score = box.conf[0].item()
            result = {
                'image_id': img_id,
                'category_id': 1,
                'bbox': [xmin, ymin, width, height],
                'score': score
            }
            results.append(result)
    # 保存检测结果
    with open(res_json_path + 'yolov8' + model_size + '_detection_results.json', 'w') as f:
        json.dump(results, f, cls=CustomJSONEncoder)
    print(res_json_path + 'yolov8' + model_size + '_detection_results.json')
    return data_size_list, infer_time_list


def coco_eval(coco, imgIds, model_size):
    accuracy_list = []  # 存储精度
    coco_pred = coco.loadRes(res_json_path + 'yolov8' + model_size + '_detection_results.json')
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    for img_id in tqdm(imgIds):
        coco_eval.params.imgIds = [img_id]
        with suppress_stdout():
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        accuracy_list.append(coco_eval.stats[0])
    return accuracy_list


def sys_run(model_size):
    imgIds = sorted(coco.getImgIds())
    imgIds.insert(0, 139)  # 在正式测试前插入一个图像用于预热程序，避免第一次装载模型等造成的冷启动时间

    data_size_list, infer_time_list = coco_ins_seg(imgIds, model_size)  # 执行所有任务的推理，保存推理结果，返回推理时间
    accuracy_list = coco_eval(coco, imgIds, model_size)  # 在推理完所有图像后统一计算精度
    del imgIds[0]  # 删除用于预热程序的样本数据
    del data_size_list[0]
    del infer_time_list[0]
    del accuracy_list[0]
    file_output = [imgIds, data_size_list, infer_time_list, accuracy_list]
    write_multi_list_to_txt(file_output, ['id', 'data_size', 'infer_time', 'accuracy'],
                            filename='yolov8' + model_size + '_data')  # 将数据保存到txt中


if __name__ == '__main__':
    coco = COCO(ann_file)
    for ann in coco.dataset['annotations']:
        ann['category_id'] = 1  # yolov8系列输出的目标类别和COCO似乎不对应，这里固定类别标签，只计算目标检测方框的精度
    coco.createIndex()
    model_list = ['n', 's', 'm', 'l', 'x']
    for model in model_list:
        sys_run(model)
