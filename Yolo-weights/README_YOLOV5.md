Current trained weights path: `../Yolo-weights/yolov5_results/runs/train/exp4/weights/last.pt` <br />
To run (from yolov5 directory): `python detect.py --weights runs/train/exp4/weights/last.pt --img 640 --conf 0.25 --source ../FULL_DATA/test/images
`<br />
To train (from yolov5 directory):` python train.py --img 640 --batch 16 --epochs 20 --data data.yaml --weights yolov5s.pt --cache    `

