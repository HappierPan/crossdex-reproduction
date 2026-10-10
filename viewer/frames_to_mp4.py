"""Convert Isaac Gym Viewer PNG recording to MP4 at the simulation's nominal 60 Hz."""
import argparse
from pathlib import Path
import cv2
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--fps',type=float,default=60);p.add_argument('--output',type=Path,default=Path('my_dagger_replay.mp4'));a=p.parse_args()
files=sorted(a.folder.glob('frame_*.png'),key=lambda f:int(f.stem.split('_')[-1]));assert files and a.fps>0,'No frames or invalid fps'
first=cv2.imread(str(files[0]));assert first is not None
size=first.shape[1::-1];w=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,size);assert w.isOpened()
for f in files:
 frame=cv2.imread(str(f));assert frame is not None and frame.shape[1::-1]==size;w.write(frame)
w.release();print(a.output)
