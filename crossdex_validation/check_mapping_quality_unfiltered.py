"""Fresh DexPilot-reference checks, not the unavailable original GRAB holdout."""
import os,sys,json
from pathlib import Path
os.chdir('/home/siyu/CrossDex/retargeting');sys.path.insert(0,os.getcwd())
import numpy as np
import torch
from manopth.manolayer import ManoLayer
from retargeting_nn_utils import EigenRetargetModel
from dex_retargeting.constants import RobotName,RetargetingType,HandType,get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.optimizer_utils import LPFilter
out=Path('/root/crossdex_validation/mapping_quality/unfiltered_executed_actions');out.mkdir(exist_ok=True)
traces=np.load('/root/crossdex_validation/trajectory_diagnostics/seed_101.npz')
RetargetingConfig.set_default_urdf_dir('/home/siyu/crossdex-deps/dex-retargeting/assets/robots/hands')
mano=ManoLayer(mano_root='../mano-models',use_pca=False,ncomps=45,flat_hand_mean=True)
rng=np.random.RandomState(20261003)
def metrics(x):
 x=np.asarray(x);return {'mean':float(x.mean()),'p95':float(np.quantile(x,.95)),'max':float(x.max())}
results={}
for hand_index,hand in enumerate(['allegro_hand','shadow_hand']):
 print('Checking',hand,flush=True)
 model=EigenRetargetModel(robot_name=hand,device='cuda')
 cfg=RetargetingConfig.load_from_file(get_default_config_path(getattr(RobotName,'allegro' if hand_index==0 else 'shadow'),RetargetingType.dexpilot,HandType.right),override={'add_dummy_free_joint':True})
 seq=cfg.build();seq.filter=None;robot=seq.optimizer.robot
 assert robot.dof_joint_names[6:]==model.robot_joint_names
 limits=robot.joint_limits[6:];lower=limits[:,0];upper=limits[:,1]
 tip_ids=[robot.get_link_index(n) for n in cfg.finger_tip_link_names];wrist=robot.get_link_index(cfg.wrist_link_name)
 def positions(q):
  robot.compute_forward_kinematics(q);inv=robot.get_link_pose_inv(wrist)
  return np.asarray([(inv@robot.get_link_pose(i))[:3,3] for i in tip_ids])
 groups={}
 samples=[]
 for env_id in range(hand_index,8,2):
  stage=np.flatnonzero(traces['lift_stage'][:,env_id]>0)
  count=min(96,int(traces['active'][:,env_id].sum()),int(stage[0])+1 if len(stage) else 999)
  samples.append(('policy_env_'+str(env_id),np.clip(traces['actions'][:count,env_id,6:15],-1,1)))
 walk=np.zeros((256,9),dtype=np.float32)
 for j in range(1,len(walk)):walk[j]=np.clip(walk[j-1]+rng.normal(0,.05,9),-.8,.8)
 samples.append(('synthetic_smooth_walk',walk))
 for name,norm in samples:
  low=model.min_values.cpu().numpy();high=model.max_values.cpu().numpy();coeff=(norm+1)*.5*(high-low)+low
  tx=torch.tensor(coeff,dtype=torch.float32,device='cuda');pose45=model.eigengrasp_to_mano_pose45(tx).float();pred=model.retarget(tx).cpu().numpy();assert np.isfinite(pred).all()
  poses=torch.zeros((len(norm),48));poses[:,3:]=pose45.cpu()
  with torch.no_grad():joints=mano(poses)[1].numpy()/1000.
  seq.reset()
  if seq.filter is not None:seq.filter.reset()
  errors=[];tip_errors=[];refs=[];excess=np.maximum(lower-pred,0)+np.maximum(pred-upper,0)
  for k,joint in enumerate(joints):
   indices=seq.optimizer.target_link_human_indices;ref=joint[indices[1]]-joint[indices[0]]
   q=seq.retarget(ref);refs.append(q[6:]);errors.append(abs(pred[k]-q[6:]))
   qnn=q.copy();qnn[6:]=np.clip(pred[k],lower,upper)
   tip_errors.append(np.linalg.norm(positions(qnn)-positions(q),axis=1))
  groups[name]={'samples':len(norm),'joint_abs_error_rad':metrics(errors),'fingertip_displacement_m':metrics(tip_errors),'raw_joint_limit_excess_rad':metrics(excess),'raw_joint_elements_outside_limits_fraction':float((excess>1e-5).mean()),'raw_samples_needing_clamp_fraction':float((excess>1e-5).any(axis=1).mean()),'per_joint_mae_rad':np.mean(errors,axis=0).tolist()}
  np.savez_compressed(out/(hand+'_'+name+'.npz'),normalized_eigengrasps=norm,nn_qpos=pred,reference_qpos=np.array(refs),fingertip_errors=np.array(tip_errors))
  print(name,json.dumps(groups[name]),flush=True)
 results[hand]={'joint_order_matches':True,'joint_names':model.robot_joint_names,'reference_filter_alpha':None,'groups':groups}
 (out/'result.json').write_text(json.dumps({'scope':'Fresh sequential unfiltered DexPilot references, policy actions truncated at forced-hold onset, and synthetic smooth walk; original GRAB labels/holdout not present on cloud. FK compares clipped NN vs reference fingertip positions in each robot wrist frame.','reference_history_caveat':'DexPilot optimization is stateful; filter disabled here to isolate filtering effects. Excludes policy actions ignored after forced-hold onset.','hands':results},indent=2)+'\n')
print('Mapping check complete',flush=True)
