# Aprendiendo a usar profiling con Nvidia Nsight en Ubuntu 22.04

## red a probar: yolactV8
install yolactV8:

```
pip install ultralytics
```

para usar gpu installa dnvo torch:

para su torch personalizado: `https://pytorch.org/get-started/locally/`

```
pip uninstall torch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

para entrenar sigue los pasos aca presentes:

```
https://github.com/ultralytics/ultralytics/blob/main/docs/modes/train.md
```

dataset: en mi drive de la tesis
obs: no corre en windows :(


## Correr nvidia Nsight

descargamso nsight compute de la pagina oficial (.run)

instalamos con `chmod +x <archivo>` seguido de `sudo ./<archivo>`

lo agregamos al path de sudo (sin sudo no sirve de nada abrirlo)

```
sudo nano /root/.bashrc
```

al final de este archvo agregamos

```
export PATH=$PATH:/usr/local/NVIDIA-Nsight-Compute

```

donde `export PATH=$PATH:/usr/local/NVIDIA-Nsight-Compute` es la direccion donde se instalo nsight

ahora para iniciarlo como sudoer, en terminal inicia como sudoer `sudo -i`y aqui abres el programa con `ncu-ui`.

# Ejecutar python con nsight

Creamos un nuevo proyecto, donde debemos indicar:

* la direccon del ejecutable (python), en este caso, la direccion de python en el env creado usando virtualenv.

* la direccion del workspace (a este repo)

* el argumento sera el programa .py a ejecutar (en este caso el que corre la evaluacion de la red eval.py)

<div align="center">
      <a href="">
     <img
      src="readme-img/nsight-run.png"
      alt="flujo de diseño Synopsys"
      style="width:70%;">
      </a>
</div>

finalmente corremos el system trace con lo que desamos ver (recuerde marcar la GPU).

# probar tensorRT version:

para esto es necesario tener instalado CUDA y CUDNN en el sistema, ademas de tensorRT, almenos en el env (usando pip)

revisar este repo: 

```
https://github.com/triple-Mu/YOLOv8-TensorRT
```

Es necesario indicar la libreria de cuddn para poder hacer el profile de la red usando tensorrt: 

``` 
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib/
```

<div align="center">
      <a href="">
     <img
      src="readme-img/profile-RT.png"
      alt="flujo de diseño Synopsys"
      style="width:70%;">
      </a>
</div>

# Resultados dataset piedra papel o tijera

se obtienen al correr los codigos:

* vanilla:
`python3 eval.py --weights='weights/best-yolo8n.pt' --imgs='datasets/data.yaml'`

* RT:
`python3 evalRT.py --engine='weights/best-yolo8n-fp16.engine' --imgs='datasets/data.yaml'`

obs: Time: average inference time per image

--- 

## Escritorio

|   yolo8n    | size MB | Time ms | mAP50 | mAP50-95 |
|-------------|---------|---------|-------|----------|
| vanilla     | 6.2     | 2.9     |0.958  |0.718     |
| fp32 (RT)   | 20.3    | 2.47    |1.0    |0.725     |
| fp16 (RT)   | 8.6     | 1.19    |0.997  |0.723     |
| int8 (RT)   | 6.1     | 0.9199  |0.997  |0.663     |


|   yolo8m    | size MB | Time ms | mAP50 | mAP50-95 |
|-------------|---------|---------|-------|----------|
| vanilla     | 103.7   | 13.5    |0.964  |0.764     |
| fp32 (RT)   | 153.5   | 11.4    |0.997  |0.746   |
| fp16 (RT)   | 55.4    | 3.88    |0.997  |0.745     |

---

## Jetson AGX Xavier

|   yolo8n    | size MB | Time ms | mAP50 | mAP50-95 |
|-------------|---------|---------|-------|----------|
| vanilla     | 6.2     | 20.3    |0.958  |0.718     |
| fp32 (RT)   | 18.1    | 20.0782 |0.997  |0.723     |
| fp16 (RT)   | 7.7     | 18.1419 |0.997  |0.723     |
| int8 (RT)   | 5       | 10.5513 |1      |0.679     |

---


## Jetson TX2

|   yolo8n    | size MB | Time ms | mAP50 | mAP50-95 |
|-------------|---------|---------|-------|----------|
| vanilla     | -     |-   |-  |-     |
| fp32 (RT)   | 26.7    | 29.574  |0.997  |0.723     |
| fp16 (RT)   | 11.5    | 22.008  |0.987  |0.714     |
| int8 (RT)   | -    | - |-     |-     |

obs: no es posible correr version vanilla ni int8 por diferencias en versiones
---

## Problemnas con Jetson TX2

* para su uso en la jetson TX2 es necesario cambiar en `models/torch_utils.py` la linea `from torchvision.ops import batched_nms` por `from torchvision.ops.boxes import batched_nms` debido a a la version de torch y torchvision que no pueden ser upgradeadas segun `https://github.com/zhiqwang/yolov5-rt-stack/issues/72`.
* dentro de evalRT.py es necesario crear la funcion asarray:
``` 
def asarray(a, dtype=None,device=None):
	if isinstance(a, torch.Tensor):
		return a
	else:
		return torch.tensor(a,dtype=dtype,device=device)
```
y buscar el uso de `torch.asarray` para remplazarlo por esa funcion, debido a que torch 1.10.0 usado por la jetson TX2 no soporta torch.asarray.

* Comentar linea 89 `cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)` ya que por alguna razon destrulle todo, esto solo hara qno se dibujen los rectangulos el lo detectado a la salida

---

# REF

* YOLOv8 `https://github.com/ultralytics/ultralytics`
* TensorRT-YOLOv8 `https://github.com/triple-Mu/YOLOv8-TensorRT`
* Entendiendo TensorRT `https://medium.com/@abhaychaturvedi_72055/understanding-nvidias-tensorrt-for-deep-learning-model-optimization-dad3eb6b26d9`
