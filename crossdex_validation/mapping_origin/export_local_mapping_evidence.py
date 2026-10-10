"""Run on the original LOCAL computer. Reads only; exports compact diagnostic evidence."""
import argparse,hashlib,json,math,pickle,gc
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('/home/siyu'));p.add_argument('--output',type=Path,default=Path('/home/siyu/crossdex_mapping_evidence'));a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);repo=a.base/'CrossDex';summary={}
pca=pickle.load((repo/'results/pca_9_grab.pkl').open('rb'))
np.savez_compressed(a.output/'pca.npz',**pca)
for hand in ['allegro_hand','shadow_hand']:
 data_path=repo/f'retargeting/dataset/retargeting_{hand}_grab_dexpilot.pkl'
 with data_path.open('rb') as f:data=pickle.load(f)
 x=np.asarray(data['mano_pose45'],dtype=np.float32);y=np.asarray(data['robot_joint_pos'],dtype=np.float32);n=len(x)
 assert x.shape==(n,45) and len(y)==n and np.isfinite(x).all() and np.isfinite(y).all()
 # Matches sklearn ShuffleSplit/test_size=.025/random_state=42 used by train_test_split.
 perm=np.random.RandomState(42).permutation(n);nval=math.ceil(n*.025);val_all=perm[:nval];train_all=perm[nval:]
 rng=np.random.RandomState(20261003);val=val_all[rng.choice(len(val_all),min(8000,len(val_all)),replace=False)];train=train_all[rng.choice(len(train_all),min(16000,len(train_all)),replace=False)]
 np.savez_compressed(a.output/(hand+'_samples.npz'),val_pose45=x[val],val_robot_joint_pos=y[val],val_indices=val,train_pose45=x[train],train_robot_joint_pos=y[train],train_indices=train)
 sx=np.zeros(45);sxx=np.zeros((45,45));pcsum=np.zeros(9);pcsq=np.zeros((9,9));pcmin=np.full(9,np.inf);pcmax=np.full(9,-np.inf)
 for start in range(0,n,32768):
  block=x[start:start+32768].astype(np.float64);sx+=block.sum(0);sxx+=block.T@block
  pc=((block-pca['D_mean'])/pca['D_std'])@pca['eigen_vectors'].T
  pcsum+=pc.sum(0);pcsq+=pc.T@pc;pcmin=np.minimum(pcmin,pc.min(0));pcmax=np.maximum(pcmax,pc.max(0))
 mean=sx/n;pcmean=pcsum/n
 text_path=a.base/('CrossDex-IsaacGym-4GB/logs/allegro-train.log' if hand=='allegro_hand' else 'CrossDex-IsaacGym-4GB/logs/other_hands_mapping_20261002_124134/shadow_train.log')
 text_lines=[]
 if text_path.exists():
  with text_path.open(errors='replace') as f:
   for line in f:
    if ('Epoch' in line and 'Loss' in line) or 'Saved best model' in line:text_lines.append(line.rstrip())
 events_dir=repo/f'retargeting/models/log_{hand}_grab_dexpilot';curves={};event_errors=[]
 try:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  for event in sorted(events_dir.glob('events.out.tfevents.*')):
   ea=EventAccumulator(str(event),size_guidance={'scalars':1000});ea.Reload()
   for tag in ['Loss/Train-epoch','Loss/Val']:
    if tag in ea.Tags()['scalars']:curves.setdefault(tag,[]).extend({'step':int(v.step),'value':float(v.value),'wall_time':float(v.wall_time),'event_file':event.name} for v in ea.Scalars(tag))
 except Exception as e:event_errors.append(str(e))
 weights=repo/f'retargeting/models/retargeting_nn_{hand}_grab_dexpilot.pth'
 summary[hand]={'dataset_path':str(data_path),'samples_total':n,'original_val_count':nval,'exported_val_count':len(val),'exported_train_count':len(train),'split_protocol':'test_size=.025, random_state=42; exported subsets sampled with independent RandomState(20261003)','joint_names':data['joint_names'],'urdf_path':str(data['urdf_path']),'robot_name':data['robot_name'],'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'pose45_mean':mean.tolist(),'pose45_covariance':(sxx/n-np.outer(mean,mean)).tolist(),'pca_coeff_mean':pcmean.tolist(),'pca_coeff_covariance':(pcsq/n-np.outer(pcmean,pcmean)).tolist(),'pca_coeff_min':pcmin.tolist(),'pca_coeff_max':pcmax.tolist(),'loss_curves':curves,'training_text_loss_lines':text_lines,'tensorboard_read_errors':event_errors,'validation_limit':'Frame-random split may share temporally adjacent frames; does not prove policy-domain generalization.'}
 (a.output/'metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(hand,'exported',len(val),'validation and',len(train),'training samples',flush=True)
 del data,x,y,perm,train_all,val_all;gc.collect()
print('Saved compact diagnostic evidence to',a.output,flush=True)
