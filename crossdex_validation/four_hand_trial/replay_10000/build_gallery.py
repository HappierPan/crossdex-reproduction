from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np,json
p=Path(__file__).resolve().parent;d=json.loads((p/'recordings.json').read_text());assert d['per_episode_outcomes_match_nonrendered'];assert all(r['matches_expected_outcome'] and r['matches_expected_length'] for r in d['camera_recordings'])
hands=['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand'];clips={};maxduration=0
for hand in hands:
 im=Image.open(p/(hand+'_success.gif'));frames=[];duration=[]
 for k in range(im.n_frames):
  im.seek(k);frames.append(im.convert('RGB').resize((640,480)));duration.append(im.info.get('duration',50))
 clips[hand]=(frames,np.cumsum(duration));maxduration=max(maxduration,sum(duration))
frames=[]
for t in range(0,maxduration,50):
 canvas=Image.new('RGB',(1280,960))
 for i,hand in enumerate(hands):
  images,ends=clips[hand];idx=min(int(np.searchsorted(ends,t,side='right')),len(images)-1);pic=images[idx].copy();draw=ImageDraw.Draw(pic);draw.rectangle((0,0,640,24),fill='black');draw.text((8,6),f'{hand} | model_10000 | successful episode',fill='white');canvas.paste(pic,((i%2)*640,(i//2)*480))
 frames.append(canvas)
frames[0].save(p/'four_hands_success.gif',save_all=True,append_images=frames[1:],duration=50,loop=0);frames[len(frames)//2].save(p/'four_hands_preview.png')
html='<!doctype html><meta charset="utf-8"><title>四手10000次模型回放</title><style>body{font-family:sans-serif;background:#10151d;color:#eee;max-width:1400px;margin:24px auto}main{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{max-width:100%;height:auto}p{line-height:1.6}</style><h1>四手10000次 PPO 最终模型</h1><p>官方 Isaac Gym 相机录像，原映射，seed101。录像结果与此前评估一致。四宫格展示四个独立成功回合，较短回合结束后保留末帧。</p><img src="four_hands_success.gif" alt="四手成功回放"><h2>单手成功与失败案例</h2><p>本次 seed101 的 Shadow64个回合全部成功，因此没有该手的失败案例。</p><main>'
for row in d['camera_recordings']:
 stem=row['hand']+'_'+row['outcome'];label='成功' if row['outcome']=='success' else '失败';html+=f'<section><h3>{row["hand"]} · {label}</h3><p>环境{row["env_id"]}，{row["actual_length"]}步</p><img src="{stem}.gif" alt="{stem}"><p><a href="{stem}.gif">单独打开录像</a></p></section>'
(p/'gallery.html').write_text(html+'</main>');print('Created four-hand montage and gallery; all rendered outcomes verified.')
