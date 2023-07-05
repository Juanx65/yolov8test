from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch
import math

import yaml
import glob 

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox
import time
import numpy as np
from shapely.geometry import Polygon

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

        val_imgs = [str(x) for x in Path(args.imgs.split("/")[0]  + "/" + val_imgs_path).rglob("*.jpg")]
        images = val_imgs
        ## ------------------------------------#
        ## codigo agreado para obtener lso MAP #
        ##-------------------------------------#
        val_labs_path = yaml_data.get("val_label", "")
        labels_path = Path(args.imgs.split("/")[0] + "/" + val_labs_path)
        ground_truth = load_ground_truth_labels(labels_path)
        #print('path: ', labels_path)
        #agregar aqui funcion que con el labels_path o yaml_data obtenga la informacion necesaria para calcular los map50 y map95 
    else:
        print('error al ingresar el dataset de validacion')
        return
    all_preds = []
    infer_time = []  # Agrega una lista para almacenar los tiempos de inferencia

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
        # Inicia el tiempo de inferencia
        start_time = time.time()
        data = Engine(tensor)
        # Calcula el tiempo de inferencia
        elapsed_time = (time.time() - start_time) * 1000 # para ms es el *1000
        infer_time.append(elapsed_time)

        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        ##############
        gt_labels, gt_boxs = get_ground_truth_label_and_box(image, ground_truth)  # Implementar esta función para obtener la etiqueta y caja de verdad de tierra
        #print("gt_boxes para primera imagen: ", gt_boxs)
        ########

        for bbox, score, label in zip(bboxes, scores, labels):
            #print("label: ", label)
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            #print("label: ",cls_id)
            color = COLORS[cls]
            #print("box predict: ", bbox)
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
            
            gt_box = None
            gt_label = None
            if gt_boxs is not None and gt_boxs:
                gt_label_indice, gt_box = bbox_mas_cercana(gt_boxs,bbox)
                #print(gt_boxs, gt_box)
                gt_boxs.remove(gt_box)#elimina el primer valor que coincida
                #print(gt_label_indice, gt_labels)
                if gt_label_indice is not None:
                    gt_label = gt_labels[gt_label_indice]
                else:
                    gt_label = None
                gt_labels.pop(gt_label_indice) # esto elimina el label segun el indice
            
            all_preds.append((gt_label, gt_box, cls_id, bbox, score))

            # Dibuja la caja del ground truth
            if gt_box is not None:
                gt_color = (0,0,255) 
                gt_box = [[int(x), int(y)] for x, y in gt_box]
                gt_box_np = np.array(gt_box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw, [gt_box_np], isClosed=True, color=gt_color, thickness=2)

        if(gt_boxs): # añadir los ground truth que no fueron detectados para el calculo de las metricas
            gt_color = (0,0,255) 
            for i, gt_box in enumerate(gt_boxs):
                gt_box = [[int(x), int(y)] for x, y in gt_box]
                gt_box_np = np.array(gt_box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw, [gt_box_np], isClosed=True, color=gt_color, thickness=2)
                all_preds.append((gt_labels[i], gt_box, None, None, None))
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)

    precisions50, recalls50 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.5)
    precisions55, recalls55 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.55)
    precisions60, recalls60 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.6)
    precisions65, recalls65 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.65)
    precisions70, recalls70 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.7)
    precisions75, recalls75 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.75)
    precisions80, recalls80 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.8)
    precisions85, recalls85 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.85)
    precisions90, recalls90 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.9)
    precisions95, recalls95 = calculate_precision_recall(ground_truth, all_preds, len(CLASSES), iou_threshold=0.95)

    #print("Precisiones (IoU=0.5):", precisions50)
    #print("Recuperaciones (IoU=0.5):", recalls50)

    mAP50 = sum(precisions50) / len(precisions50)
    mAP55 = sum(precisions55) / len(precisions55)
    mAP60 = sum(precisions60) / len(precisions60)
    mAP65 = sum(precisions65) / len(precisions65)
    mAP70 = sum(precisions70) / len(precisions70)
    mAP75 = sum(precisions75) / len(precisions75)
    mAP80 = sum(precisions80) / len(precisions80)
    mAP85 = sum(precisions85) / len(precisions85)
    mAP90 = sum(precisions90) / len(precisions90)
    mAP95 = sum(precisions95) / len(precisions95)
    mAP50_95= mAP50_95 = (mAP50 + mAP55 + mAP60 + mAP65 + mAP70 + mAP75 + mAP80 + mAP85 + mAP90 + mAP95) / 10

    print("mAP50:", mAP50)
    print("mAP50-95:", mAP50_95)
    avg_infer_time = sum(infer_time) / len(infer_time)
    print(f"Tiempo promedio de inferencia por imagen: {avg_infer_time:.4f} ms")

def load_ground_truth_labels(labels_path):
    ground_truth = {}

    for label_file in glob.glob(str(labels_path / "*.txt")):
        with open(label_file, "r") as f:
            filename = Path(label_file).stem
            ground_truth[filename] = []
            for line in f:
                parts = line.strip().split()
                cls = parts[0]
                points = list(map(float, parts[1:]))
                ground_truth[filename].append((cls, points))
    return ground_truth

def get_ground_truth_label_and_box(image, ground_truth):
    filename = Path(image).stem
    if filename not in ground_truth:
        return None, None

    gt_labels_boxes = ground_truth[filename]
    if not gt_labels_boxes:
        return None, None

    bboxs = []
    labels = []
    for cls, coords in gt_labels_boxes:
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]  # Reorganizamos x e y en pares
        scaled_points = [[int(x*640), int(y*640)] for x, y in points]
        #print(scaled_points)
        bboxs.append(scaled_points)
        labels.append(int(cls))
    return labels, bboxs



def centroid(poly):
    # Calcular el centroide de un polígono.
    # Polígono es una lista de pares de coordenadas [(x1, y1), (x2, y2), ...].
    x_coords = [p[0] for p in poly]
    y_coords = [p[1] for p in poly]
    centroid_x = sum(x_coords) / len(poly)
    centroid_y = sum(y_coords) / len(poly)
    return centroid_x, centroid_y

def bbox_mas_cercana(ground_truth, prediccion):
    if ground_truth is not None:
        distancia_minima = math.inf
        bbox_mas_cercana = None
        indice_label = 0
        # Calcular el centroide del rectángulo de predicción
        pred_centroid = centroid([(prediccion[0], prediccion[1]), 
                                  (prediccion[0], prediccion[3]),
                                  (prediccion[2], prediccion[1]),
                                  (prediccion[2], prediccion[3])])
        for i, poly in enumerate(ground_truth):
            # Calcular el centroide del polígono de ground truth
            gt_centroid = centroid(poly)
            # Calcular la distancia euclidiana entre los centroides
            distancia = np.sqrt((gt_centroid[0] - pred_centroid[0])**2 + 
                                (gt_centroid[1] - pred_centroid[1])**2)
            # Actualizar la bbox más cercana si la distancia actual es menor que la distancia mínima anterior
            if distancia < distancia_minima:
                distancia_minima = distancia
                bbox_mas_cercana = poly
                indice_label = i
        return indice_label, bbox_mas_cercana
    else:
        return None, None

def calculate_precision_recall(gt, preds, n_classes, iou_threshold=0.5):
    gt_labels = []
    gt_boxes = []
    for key in gt:
        for data in gt[key]:
            cls = data[0]
            coords = data[1:]  # Asumiendo que los puntos del polígono vienen después del label en tu data
            gt_labels.append(cls)
            gt_boxes.append(coords)

    precisions = []
    recalls = []
    for cls in range(n_classes):
        tp, fp = 0, 0
        y_true = []
        y_pred = []

        for gt_label, gt_box, pred_label, pred_box, pred_score in preds:
            if gt_label == cls and pred_label == cls:
                iou = calculate_iou(gt_box, pred_box)
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

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def calculate_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon([(box2[0], box2[1]), (box2[2], box2[1]), (box2[2], box2[3]), (box2[0], box2[3])])  # Convertimos las coordenadas del rectangulo a un poligono
    
    if not poly1.is_valid or not poly2.is_valid:
        print('Invalid polygon')
        return 0

    inter_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - inter_area

    return inter_area / union_area

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