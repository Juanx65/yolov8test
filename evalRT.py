from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch
import os
import math

import yaml
from sklearn.metrics import average_precision_score
import glob 

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]
    save_path = Path(args.out_dir)
    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

     # Verifica si args.imgs es un archivo YAML
    if args.imgs.endswith(".yaml"):
        with open(args.imgs, "r") as file:
            yaml_data = yaml.safe_load(file)

        val_imgs_path = yaml_data.get("val", "")
        if not val_imgs_path:
            raise ValueError("No se encontró el conjunto de validación en el archivo YAML")

        val_imgs = [str(x) for x in Path('datasets/'+val_imgs_path).rglob("*.jpg")]
        images = val_imgs
        ## ------------------------------------#
        ## codigo agreado para obtener lso MAP #
        ##-------------------------------------#
        val_labs_path = yaml_data.get("val_label", "")
        labels_path = Path('datasets/'+ val_labs_path)
        ground_truth = load_ground_truth_labels(labels_path)
        #print('path: ', labels_path)
        #agregar aqui funcion que con el labels_path o yaml_data obtenga la informacion necesaria para calcular los map50 y map95 
    else:
        print('error al ingresar el dataset de validacion')
        return
    all_preds = []
    for i, image in enumerate(images):
        save_image = save_path / image.split('/valid/images/')[1]
        #print(save_image)
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        ##############
        gt_labels, gt_boxs = get_ground_truth_label_and_box(image, ground_truth)  # Implementar esta función para obtener la etiqueta y caja de verdad de tierra
        #print("gt_boxes para primera imagen: ", gt_boxs)
        ########

        for j, (bbox, score, label) in enumerate( zip(bboxes, scores, labels)):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            #print("box predict: ", bbox)
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
            
            gt_label_indice, gt_box = bbox_mas_cercana(gt_boxs,bbox)
           #print(gt_label_indice, gt_labels)
            gt_label = gt_labels[gt_label_indice]
            
            all_preds.append((gt_label, gt_box, label, bbox, score))

            # Dibuja la caja del ground truth
            if gt_box is not None:
                gt_color = (0,0,255) 
                x1, y1, x2, y2 = [int(coord) for coord in gt_box]
                #print("box gt: ", x1, y1, x2, y2)
                cv2.rectangle(draw, (x1, y1), (x2, y2), gt_color, 2)
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)

    precisions50, recalls50 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.5)
    precisions95, recalls95 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.95)

    print("Precisiones (IoU=0.5):", precisions50)
    print("Recuperaciones (IoU=0.5):", recalls50)
    print("Precisiones (IoU=0.95):", precisions95)
    print("Recuperaciones (IoU=0.95):", recalls95)

    mAP50 = sum(precisions50) / len(precisions50)
    mAP95 = sum(precisions95) / len(precisions95)

    print("mAP50:", mAP50)
    print("mAP95:", mAP95)

def load_ground_truth_labels(labels_path):
    ground_truth = {}

    for label_file in glob.glob(str(labels_path / "*.txt")):
        with open(label_file, "r") as f:
            filename = Path(label_file).stem
            ground_truth[filename] = []
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                ground_truth[filename].append((cls, x, y, w, h))
    #print(ground_truth)
    return ground_truth

def calculate_precision_recall(gt, preds, n_classes, iou_threshold=0.5):
    #print('preds: ', preds)
    gt_labels = []
    gt_boxes = []
    for key in gt:
        for cls, x, y, w, h in gt[key]:
            gt_labels.append(cls)
            gt_boxes.append((x, y, w, h))

    #print(gt_boxes)
    precisions = []
    recalls = []
    for cls in range(n_classes):
        tp, fp = 0, 0
        y_true = []
        y_pred = []

        for gt_label, gt_box, pred_label, pred_box, pred_score in preds:
            if gt_label == cls and pred_label == cls:
                iou = calculate_iou( gt_box, pred_box)
                #print('iou, gt, pred: ', iou,', ', gt_box, ', ', pred_box)
                if iou >= iou_threshold:
                    tp += 1
                else:
                    fp += 1

                y_true.append(1)
                y_pred.append(pred_score)

            elif gt_label == cls:
                y_true.append(1)
                y_pred.append(0)

            elif pred_label == cls:
                y_true.append(0)
                y_pred.append(pred_score)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / len([x for x in gt_labels if x == cls]) if len([x for x in gt_labels if x == cls]) > 0 else 0

        #print('precision, recall: ', precision,recall )

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2

    print("box gt: ", x1, y1, x2, y2)
    print("box pr: ", X1, Y1, X2, Y2)

    xi1 = max(x1, X1)
    yi1 = max(y1, Y1)
    xi2 = min(x2, X2)
    yi2 = min(y2, Y2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (X2 - X1) * (Y2 - Y1)
    #print('box areas: ', box1_area, ' , ', box2_area)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def get_ground_truth_label_and_box(image, ground_truth):
    filename = Path(image).stem
    #print('filname: ', filename)
    if filename not in ground_truth:
        return None, None

    gt_labels_boxes = ground_truth[filename]
    if not gt_labels_boxes:
        return None, None
    
    bboxs = []
    labels = []
    for cls, x, y, w, h in gt_labels_boxes:
        bboxs.append([ int((x - w / 2)*640), int((y - h / 2)*640), int((x + w / 2)*640), int((y + h / 2)*640)])
        labels.append(int(cls))
        #print('cls, bbox: ', cls,bbox)
    return labels, bboxs

def bbox_mas_cercana(ground_truth, prediccion):
    distancia_minima = math.inf
    bbox_mas_cercana = None
    indice_label = 0
    for i, bbox in enumerate(ground_truth):
        # Calcular la distancia euclidiana entre la predicción y cada bbox de ground truth
        distancia = math.sqrt((bbox[0] - prediccion[0])**2 + (bbox[1] - prediccion[1])**2 + 
                              (bbox[2] - prediccion[2])**2 + (bbox[3] - prediccion[3])**2)
        # Actualizar la bbox más cercana si la distancia actual es menor que la distancia mínima anterior
        if distancia < distancia_minima:
            distancia_minima = distancia
            bbox_mas_cercana = bbox
            indice_label = i
    return indice_label, bbox_mas_cercana

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)