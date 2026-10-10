import os,sys,json,xml.etree.ElementTree as ET
from pathlib import Path
os.chdir('/home/siyu/CrossDex/retargeting');sys.path.insert(0,os.getcwd())
import numpy as np,torch
from retargeting_nn_utils import EigenRetargetModel
from dex_retargeting.robot_wrapper import RobotWrapper
from manopth.manolayer import ManoLayer
tr=np.load('/root/crossdex_validation/trajectory_diagnostics/seed_101.npz');out=Path('/root/crossdex_validation/mapping_origin');runtime=json.loads((out/'runtime_chain.json').read_text());rng=np.random.RandomState(42)
mano=ManoLayer(mano_root='../mano-models',use_pca=False,ncomps=45,flat_hand_mean=True)
result={}
def stats(a):
 a=np.asarray(a);return {'mean':float(a.mean()),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
for i,hand in enumerate(['allegro_hand','shadow_hand']):
 m=EigenRetargetModel(robot_name=hand,device='cuda');stand=RobotWrapper(m.robot_urdf_path);arm=RobotWrapper('/home/siyu/CrossDex/robot_randomization/urdf_origin/rm65_'+('allegro' if i==0 else 'shadow')+'_right.urdf')
 names=m.robot_joint_names;stand_idx=[stand.get_joint_index(n) for n in names];arm_idx=[arm.get_joint_index(n) for n in names]
 limits=stand.joint_limits[stand_idx];actual=np.stack([runtime[hand]['limits_lower'],runtime[hand]['limits_upper']],axis=1)
 tips=['link_15.0_tip','link_3.0_tip','link_7.0_tip','link_11.0_tip'] if i==0 else ['thtip','fftip','mftip','rftip','lftip'];w='wrist' if i==0 else 'ee_link'
 def pos(robot,q):
  robot.compute_forward_kinematics(q);inv=robot.get_link_pose_inv(robot.get_link_index(w));return np.asarray([(inv@robot.get_link_pose(robot.get_link_index(t)))[:3,3] for t in tips])
 errs=[]
 for k in range(64):
  q=rng.uniform(limits[:,0],limits[:,1]);qs=stand.q0.copy();qa=arm.q0.copy();qs[stand_idx]=q;qa[arm_idx]=q;errs.extend(np.linalg.norm(pos(stand,qs)-pos(arm,qa),axis=1))
 valid=tr['active'][:,i::2].copy();valid[0]=False;valid[1:]&=tr['lift_stage'][:-1,i::2]==0
 raw_actions=tr['actions'][:,:,6:15][:,i::2][valid];norm=np.clip(raw_actions,-1,1);low=m.min_values.cpu().numpy();high=m.max_values.cpu().numpy();coeff=(norm+1)*.5*(high-low)+low
 selected=coeff[rng.choice(len(coeff),256,replace=False)];x=torch.tensor(selected,dtype=torch.float32,device='cuda');pose=m.eigengrasp_to_mano_pose45(x).float();blocks=pose.reshape(-1,15,3);angle=blocks.norm(dim=-1);wrapped_angle=(angle+np.pi)%(2*np.pi)-np.pi;wrapped=(blocks*torch.where(angle>1e-8,wrapped_angle/angle,torch.ones_like(angle))[:,:,None]).reshape(-1,45)
 with torch.no_grad():
  original_nn=m.retarget_from_pose45(pose).cpu().numpy();wrapped_nn=m.retarget_from_pose45(wrapped).cpu().numpy()
  def human(p):
   full=torch.zeros((len(p),48));full[:,3:]=p.cpu();return mano(full)[1].numpy()
  original_human=human(pose);wrapped_human=human(wrapped)
 result[hand]={'standalone_vs_simulator_hand_limits_max_abs_difference_rad':float(abs(limits-actual).max()),'standalone_vs_arm_attached_fk_fingertip_difference_m':stats(errs),'policy_samples_before_forced_hold':len(coeff),'raw_action_components_outside_minus1_plus1_fraction':float((abs(raw_actions)>1).mean()),'clipped_action_components_at_bounds_fraction':float((abs(norm)>=.999).mean()),'pca_coefficient_norm':stats(np.linalg.norm(coeff,axis=1)),'sampled_pose_joint_axis_angle_norm_rad':stats(angle.cpu().numpy()),'sampled_pose_joints_over_pi_fraction':float((angle>np.pi).float().mean()),'equivalent_axis_angle_wrap_human_keypoints_max_difference_mm':float(abs(original_human-wrapped_human).max()),'equivalent_axis_angle_wrap_nn_joint_difference_rad':stats(abs(original_nn-wrapped_nn)),'representation_check_scope':'Wrapping is a diagnostic only; no preprocessing or model weights changed. Does not establish training input coverage.'}
 print(hand,json.dumps(result[hand]),flush=True)
(out/'assets_and_input.json').write_text(json.dumps(result,indent=2)+'\n')
