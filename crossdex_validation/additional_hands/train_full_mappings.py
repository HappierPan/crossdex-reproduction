"""Generate both missing official DexPilot datasets and train original 200-epoch MLPs."""
import argparse,os,sys,json,hashlib,time,fcntl
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('/root/crossdex_validation/additional_hands/full_run'));a=p.parse_args();source=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.chdir('/home/siyu/CrossDex/retargeting');sys.path.insert(0,os.getcwd())
import numpy as np,torch
from manopth.manolayer import ManoLayer
from dex_retargeting.constants import RobotName,RetargetingType,HandType,get_default_config_path,ROBOT_NAME_MAP
from dex_retargeting.retargeting_config import RetargetingConfig
import train_retargeting_nn as official
from retargeting_nn_utils import RetargetingNN
np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
RetargetingConfig.set_default_urdf_dir('/home/siyu/crossdex-deps/dex-retargeting/assets/robots/hands')
metadata=json.loads(source.with_suffix('.json').read_text());assert hashlib.sha256(source.read_bytes()).hexdigest()==metadata['archive_sha256']
inputs=np.load(source);poses=inputs['mano_pose45'];assert poses.shape==(metadata['frames'],45) and len(poses)>1000000 and np.isfinite(poses).all();assert hashlib.sha256(poses.tobytes()).hexdigest()==metadata['pose45_sha256']
status={'stage':'starting','input':str(source),'frames':len(poses),'hands':{},'ppo_started':False,'production_weights_written':False}
def save(**kw):
 status.update(kw);status['updated_unix']=time.time();tmp=out/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(out/'status.json')
mano=ManoLayer(mano_root='../mano-models',use_pca=False,ncomps=45,flat_hand_mean=True)
try:
 for enum in [RobotName.ability,RobotName.svh]:
  hand=ROBOT_NAME_MAP[enum];dst=out/hand;dst.mkdir(exist_ok=True);save(stage='pair_generation',active_hand=hand,generated_frames=0)
  cfg=RetargetingConfig.load_from_file(get_default_config_path(enum,RetargetingType.dexpilot,HandType.right),override={'add_dummy_free_joint':True});seq=cfg.build();dim=len(seq.joint_names)-6;idx=seq.optimizer.target_link_human_indices
  label_file=dst/'robot_joint_pos.npy';complete=dst/'pair_generation_complete.json'
  if complete.exists():
   pairinfo=json.loads(complete.read_text());assert pairinfo['pose45_sha256']==metadata['pose45_sha256'];labels=np.load(label_file,mmap_mode='r');assert labels.shape==(len(poses),dim)
  else:
   labels=np.lib.format.open_memmap(label_file,mode='w+',dtype=np.float32,shape=(len(poses),dim))
   for start in range(0,len(poses),128):
    stop=min(start+128,len(poses));mp=torch.zeros((stop-start,48));mp[:,3:]=torch.tensor(poses[start:stop])
    with torch.no_grad():human=mano(mp)[1].numpy()/1000.
    for k,h in enumerate(human):labels[start+k]=seq.retarget(h[idx[1]]-h[idx[0]])[6:]
    assert np.isfinite(labels[start:stop]).all()
    if start%4096==0:labels.flush();save(generated_frames=stop);print(hand,'paired',stop,'/',len(poses),flush=True)
   labels.flush();complete.write_text(json.dumps({'pose45_sha256':metadata['pose45_sha256'],'joint_names':seq.joint_names[6:],'frames':len(poses),'method':'Official ordered continuous SeqRetargeting and default LPFilter; no label clipping.'},indent=2)+'\n')
  data={'mano_pose45':poses,'robot_joint_pos':labels,'robot_name':hand,'joint_names':seq.joint_names[6:],'urdf_path':str(cfg.urdf_path)}
  official.load_dataset=lambda *args,**kwargs:data
  trainer=official.Trainer(robot_name=enum,device='cuda');stem='retargeting_nn_'+hand+'_grab_dexpilot';trainer.save_config(dst/(stem+'.yaml'))
  save(stage='mapping_training',active_hand=hand,epochs=200,generated_frames=len(poses))
  # Reuse official architecture, random 2.5% split (42), Adam and 200-epoch recipe.
  torch.manual_seed(42);trainer.train(batch_size=256,num_epochs=200,save_path=dst/(stem+'.pth'),log_dir=str(dst/'tensorboard'))
  model=RetargetingNN(dim).cuda();model.load_state_dict(torch.load(dst/(stem+'.pth')));model.eval();val=trainer.val_dataset
  with torch.no_grad():
   pred=np.concatenate([model(val.tensors[0][i:i+1024].cuda()).cpu().numpy() for i in range(0,len(val),1024)])
  target=val.tensors[1].numpy();error=pred-target;assert np.isfinite(pred).all();rng=np.random.RandomState(42);sample=rng.choice(len(pred),min(512,len(pred)),replace=False);robot=seq.optimizer.robot;wrist=robot.get_link_index(cfg.wrist_link_name);tips=[robot.get_link_index(n) for n in cfg.finger_tip_link_names];limits=robot.joint_limits[6:]
  def fk(joints):
   q=np.zeros(robot.dof);q[6:]=np.clip(joints,limits[:,0],limits[:,1]);robot.compute_forward_kinematics(q);inv=robot.get_link_pose_inv(wrist);return np.array([(inv@robot.get_link_pose(i))[:3,3] for i in tips])
  tiperror=np.concatenate([np.linalg.norm(fk(pred[i])-fk(target[i]),axis=1)*1000 for i in sample])
  report={'hand':hand,'train_frames':len(trainer.train_dataset),'validation_frames':len(val),'validation_mse_rad2':float((error**2).mean()),'validation_rmse_rad':float(np.sqrt((error**2).mean())),'validation_mae_rad':float(abs(error).mean()),'sampled_mean_fingertip_error_mm':float(tiperror.mean()),'sampled_p95_fingertip_error_mm':float(np.quantile(tiperror,.95)),'original_frame_random_split':True,'limitation':'Adjacent frames may occur across training and validation. Does not establish PPO-domain coverage or grasp performance.','weights':str(dst/(stem+'.pth')),'weight_sha256':hashlib.sha256((dst/(stem+'.pth')).read_bytes()).hexdigest()}
  (dst/'quality.json').write_text(json.dumps(report,indent=2)+'\n');status['hands'][hand]=report;save()
 save(stage='complete_independent_mapping_training')
except Exception as error:
 save(stage='failed',error=str(error));raise
