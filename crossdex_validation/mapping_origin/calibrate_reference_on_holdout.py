import os,sys,json
from pathlib import Path
os.chdir('/home/siyu/CrossDex/retargeting');sys.path.insert(0,os.getcwd())
import numpy as np,torch
from manopth.manolayer import ManoLayer
from retargeting_nn_utils import EigenRetargetModel
from dex_retargeting.constants import RobotName,RetargetingType,HandType,get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
out=Path('/root/crossdex_validation/mapping_origin');trace=np.load('/root/crossdex_validation/trajectory_diagnostics/seed_101.npz');rng=np.random.RandomState(20261003)
RetargetingConfig.set_default_urdf_dir('/home/siyu/crossdex-deps/dex-retargeting/assets/robots/hands')
mano=ManoLayer(mano_root='../mano-models',use_pca=False,ncomps=45,flat_hand_mean=True)
def stat(x):
 a=np.array(x);return {'mean':float(a.mean()),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
result={}
evidence=Path('/root/crossdex_validation/mapping_origin/crossdex_mapping_evidence')
for hindex,hand in enumerate(['allegro_hand','shadow_hand']):
 m=EigenRetargetModel(robot_name=hand,device='cuda')
 cfg=RetargetingConfig.load_from_file(get_default_config_path(RobotName.allegro if hindex==0 else RobotName.shadow,RetargetingType.dexpilot,HandType.right),override={'add_dummy_free_joint':True})
 seq=cfg.build();seq.filter=None;robot=seq.optimizer.robot;op=seq.optimizer
 assert robot.dof_joint_names[6:]==m.robot_joint_names
 tips=[robot.get_link_index(n) for n in cfg.finger_tip_link_names];wrist=robot.get_link_index(cfg.wrist_link_name);limits=robot.joint_limits
 def pos(q):
  robot.compute_forward_kinematics(q);inv=robot.get_link_pose_inv(wrist);return np.array([(inv@robot.get_link_pose(i))[:3,3] for i in tips])
 active=trace['active'][:,hindex::2];stage=trace['lift_stage'][:,hindex::2];valid=active.copy();valid[0]=False;valid[1:]&=stage[:-1]==0
 actions=trace['actions'][:,:,6:15][:,hindex::2][valid];actions=np.clip(actions,-1,1);actions=actions[rng.choice(len(actions),64,replace=False)]
 low=m.min_values.cpu().numpy();high=m.max_values.cpu().numpy();policy=(actions+1)*.5*(high-low)+low
 original=np.load(evidence/(hand+'_samples.npz'))
 selection=rng.choice(len(original['val_pose45']),64,replace=False)
 groups={'original_holdout_pose45':original['val_pose45'][selection]}
 rows={}
 for name,coeff in groups.items():
  tx=torch.tensor(coeff,dtype=torch.float32,device='cuda');pose=tx.float()
  with torch.no_grad():pred=m.retarget_from_pose45(tx).cpu().numpy();mp=torch.zeros((len(coeff),48));mp[:,3:]=pose.cpu();human=mano(mp)[1].numpy()/1000.
  ref_error=[];best_error=[];raw_mae=[];clip_mae=[];excess=[];cost_nn=[];cost_ref=[];cost_best=[];root=[];pred_max=[];nn_label_error=[];default_label_error=[];best_label_error=[]
  for k,human_joints in enumerate(human):
   seq.reset();op.projected[:]=False
   idx=op.target_link_human_indices;ref=human_joints[idx[1]]-human_joints[idx[0]]
   qref=seq.retarget(ref);qnn=qref.copy();qnn[6:]=np.clip(pred[k],limits[6:,0],limits[6:,1])
   objective=op.get_objective_function(ref.astype(np.float32),np.array([]),qref[op.idx_pin2target])
   cn=objective(qnn[op.idx_pin2target],np.array([]));cr=objective(qref[op.idx_pin2target],np.array([]))
   alt=op.retarget(ref,np.array([]),np.clip(qnn[op.idx_pin2target],seq.joint_limits[:,0],seq.joint_limits[:,1]))
   qa=np.zeros(robot.dof);qa[op.idx_pin2target]=alt
   # Re-evaluate all candidates against one identical projected target.
   objective=op.get_objective_function(ref.astype(np.float32),np.array([]),qref[op.idx_pin2target])
   cn=objective(qnn[op.idx_pin2target],np.array([]));cr=objective(qref[op.idx_pin2target],np.array([]));ca=objective(qa[op.idx_pin2target],np.array([]))
   qb=qa if ca<cr else qref
   ref_error.extend(np.linalg.norm(pos(qnn)-pos(qref),axis=1));best_error.extend(np.linalg.norm(pos(qnn)-pos(qb),axis=1))
   qlabel=qref.copy();qlabel[6:]=original['val_robot_joint_pos'][selection[k]]
   nn_label_error.extend(np.linalg.norm(pos(qnn)-pos(qlabel),axis=1));default_label_error.extend(np.linalg.norm(pos(qref)-pos(qlabel),axis=1));best_label_error.extend(np.linalg.norm(pos(qb)-pos(qlabel),axis=1))
   raw_mae.extend(abs(pred[k]-qref[6:]));clip_mae.extend(abs(qnn[6:]-qref[6:]));excess.extend(np.maximum(limits[6:,0]-pred[k],0)+np.maximum(pred[k]-limits[6:,1],0));cost_nn.append(cn);cost_ref.append(cr);cost_best.append(min(cr,ca))
  rows[name]={'samples':len(coeff),'clipped_nn_vs_default_reference_tip_error_m':stat(ref_error),'clipped_nn_vs_best_two_initializations_tip_error_m':stat(best_error),'raw_joint_abs_error_rad':stat(raw_mae),'clipped_joint_abs_error_rad':stat(clip_mae),'raw_limit_excess_rad':stat(excess),'raw_joint_elements_needing_clamp_fraction':float((np.array(excess)>1e-5).mean()),'same_dexpilot_cartesian_objective_nn':stat(cost_nn),'same_dexpilot_cartesian_objective_default_reference':stat(cost_ref),'same_dexpilot_cartesian_objective_best_reference':stat(cost_best),'nn_vs_original_label_tip_error_m':stat(nn_label_error),'fresh_default_reference_vs_original_label_tip_error_m':stat(default_label_error),'fresh_best_reference_vs_original_label_tip_error_m':stat(best_label_error),'pose45_abs_max_rad':float(abs(pose.cpu().numpy()).max())}
  print(hand,name,json.dumps(rows[name]),flush=True)
 result[hand]=rows
 (out/'reference_calibration.json').write_text(json.dumps({'method':'Independent cold-start DexPilot per sample; no output LP filter; projection flags reset; additional NN-initialized optimizer result; same objective evaluated across candidates. Policy actions sampled from all envs before forced-hold onset, excluding initial frame. Scaling contracts PCA coefficients towards PCA zero, not action midpoint. No original heldout GRAB dataset available.','hands':result},indent=2)+'\n')
print('Controlled checks complete',flush=True)
