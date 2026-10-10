"""Convert existing GIF timelines to20fps MP4 using the installed OpenCV encoder."""
from pathlib import Path
import json,math,zipfile
import numpy as np
from PIL import Image
import cv2
p=Path('/root/crossdex_validation/single_object_dagger/expanded/replay');fps=20;report=[]
for source in sorted(p.glob('*.gif')):
 im=Image.open(source);frames=[];durations=[]
 for i in range(im.n_frames):
  im.seek(i);frames.append(im.convert('RGB'));durations.append(im.info.get('duration',50))
 ends=np.cumsum(durations);total=ends[-1]/1000;count=int(math.ceil(total*fps));target=source.with_suffix('.mp4');size=frames[0].size
 w=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,size);assert w.isOpened(),target
 for i in range(count):
  idx=min(int(np.searchsorted(ends,i*1000/fps,side='right')),len(frames)-1);rgb=np.array(frames[idx]);w.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
 w.release();cap=cv2.VideoCapture(str(target));assert cap.isOpened();decoded=0
 while True:
  ok,f=cap.read()
  if not ok:break
  assert f.shape[1::-1]==size;decoded+=1
 cap.release();assert decoded==count and abs(count/fps-total)<=1/fps+1e-6
 report.append({'source':str(source),'mp4':str(target),'codec':'MPEG-4 Part2 (mp4v)','fps':fps,'frames':count,'source_seconds':total,'video_seconds':count/fps,'decoded_frames_verified':decoded,'bytes':target.stat().st_size});print(target.name,count,'frames',round(total,2),'seconds',flush=True)
(p/'mp4_conversion.json').write_text(json.dumps(report,indent=2)+'\n')
html=(p/'gallery.html').read_text()
for row in report:
 stem=Path(row['source']).stem
 html=html.replace(f'<img src="{stem}.gif"',f'<p><a href="{stem}.mp4" download>下载 MP4 视频</a></p><img src="{stem}.gif"')
(p/'gallery.html').write_text(html)
with zipfile.ZipFile(p/'four_hand_mp4_videos.zip','w',zipfile.ZIP_STORED) as z:
 for row in report:z.write(row['mp4'],Path(row['mp4']).name)
print('All8 MP4 clips decoded successfully. MP4 download links added to gallery.')
