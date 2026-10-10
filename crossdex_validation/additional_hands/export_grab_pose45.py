"""Run on the local data machine. Export complete GRAB inputs only; no robot labels needed."""
import argparse,json,pickle,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--grab-dir',type=Path,default=Path('/home/siyu/CrossDex/GRAB/hand_dataset'));p.add_argument('--paired',type=Path,default=Path('/home/siyu/CrossDex/retargeting/dataset/retargeting_allegro_hand_grab_dexpilot.pkl'));p.add_argument('--output',type=Path,default=Path('crossdex_grab_pose45.npz'));a=p.parse_args()
if all((a.grab_dir/f's{i}.pkl').exists() for i in range(1,11)):
 poses=[];ids=[];clips=[];sid=0;offset=0
 for subject in range(1,11):
  src=a.grab_dir/f's{subject}.pkl'
  with src.open('rb') as f:sequences=pickle.load(f)
  for seq in sequences:
   x=np.asarray(seq,dtype=np.float32);assert x.ndim==2 and x.shape[1]==45
   poses.append(x);ids.append(np.full(len(x),sid,dtype=np.int32));clips.append({'sequence_id':sid,'subject':subject,'start':offset,'frames':len(x)});sid+=1;offset+=len(x)
 x=np.concatenate(poses);sequence_id=np.concatenate(ids);source='Original GRAB s1..s10, concatenated in official order'
else:
 with a.paired.open('rb') as f:d=pickle.load(f)
 x=np.asarray(d['mano_pose45'],dtype=np.float32);sequence_id=np.full(len(x),-1,dtype=np.int32);clips=[];source=str(a.paired)+' mano_pose45; original sequence boundaries unavailable'
assert x.ndim==2 and x.shape[1]==45 and len(x)>1000000 and np.isfinite(x).all()
np.savez_compressed(a.output,mano_pose45=x,sequence_id=sequence_id)
meta={'frames':len(x),'dimension':45,'source':source,'sequence_boundaries_available':bool(clips),'sequences':clips,'pose45_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'archive_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print('Exported',a.output,len(x),'frames;',round(a.output.stat().st_size/1024**2,1),'MiB')
