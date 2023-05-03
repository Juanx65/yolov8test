from ultralytics import YOLO
import os
import argparse

def main(args:argparse.Namespace) -> None:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_directory,args.imgs)
    model = YOLO(args.weights)

    #model.track(source='Data1.mp4',show=True, stream=False, verbose=False) # detectar en el video de testeo

    model.val(data=data_path, verbose=False) # evaluar

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Engine file',default='weights/best.pt')
    parser.add_argument('--imgs', type=str, help='Images file', default='datasets/data.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)