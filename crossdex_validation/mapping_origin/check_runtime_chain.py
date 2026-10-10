import os,sys,json
from pathlib import Path
os.chdir('/home/siyu/CrossDex/rl');sys.path[:0]=[os.getcwd(),'/home/siyu/CrossDex']
from isaacgym import gymapi
import torch,numpy as np
from omegaconf import OmegaConf
import isaacgymenvs
import run_ppo_multidex
cfg=OmegaConf.create(json.loads(Path('/root/autodl-tmp/crossdex_experiments/trial_8192_20261002_234722/runs/077_rubiks_cube_2026-10-02_23-47-34/config.json').read_text()))
cfg.num_envs=cfg.task.env.numEnvs=2;cfg.test=cfg.task.is_test=cfg.train.params.test=True
cfg.headless=True;cfg.capture_video=False;cfg.force_render=False
tr=np.load('/root/crossdex_validation/trajectory_diagnostics/seed_101.npz')
env=isaacgymenvs.make(42,cfg.task_name,2,cfg.sim_device,cfg.rl_device,0,True,False,False,False,cfg)
result={}
try:
 for i,hand in enumerate(env.hand_names):
  idx=env.per_robot_dof_indices[i][env.hand_dof_indices[i]].cpu().numpy();order=np.array(env.retarget2isaacs[i]);model=env.retargeting_models[i]
  low=env.agg_robot_dof_lower_limits[idx].cpu().numpy();high=env.agg_robot_dof_upper_limits[idx].cpu().numpy()
  norm=np.clip(tr['actions'][:,:,6:15][:,i::2],-1,1);shape=norm.shape
  inp=torch.tensor(norm.reshape(-1,9),device='cuda');coeff=(inp+1)*.5*(model.max_values-model.min_values)+model.min_values
  with torch.no_grad():raw=model.retarget(coeff).cpu().numpy().reshape(shape[:2]+(model.robot_dim,))[:,:,order]
  # Recorded targets use the same interleaved aggregate environment layout.
  target=tr['targets'][:,:,idx];pred=.1*np.clip(raw[1:],low,high)+.9*target[:-1]
  valid=tr['active'][1:,i::2] & (tr['lift_stage'][:-1,i::2]==0)
  error=abs(pred-target[1:])[valid]
  pca_np=(coeff.cpu().numpy()@model.principal_vectors.cpu().numpy())*model.D_std.cpu().numpy()+model.D_mean.cpu().numpy()
  pca_t=model.eigengrasp_to_mano_pose45(coeff).cpu().numpy()
  result[hand]={'runtime_joint_names':env.hand_dof_names[i],'network_joint_names':model.robot_joint_names,'retarget2isaac':order.tolist(),'mapped_joint_count':len(order),'joint_name_mapping_valid':all(model.robot_joint_names[order[j]]==n for j,n in enumerate(env.hand_dof_names[i])),'pca_numpy_vs_torch_max_abs_error':float(abs(pca_np-pca_t).max()),'recorded_target_comparison_samples':int(valid.sum()),'reconstructed_vs_recorded_hand_target_max_abs_error_rad':float(error.max()),'reconstructed_vs_recorded_hand_target_mean_abs_error_rad':float(error.mean()),'raw_joint_limit_excess_fraction':float(((raw<low-1e-5)|(raw>high+1e-5))[tr['active'][:,i::2]].mean()),'limits_lower':low.tolist(),'limits_upper':high.tolist()}
 Path('/root/crossdex_validation/mapping_origin/runtime_chain.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
finally:env.gym.destroy_sim(env.sim)
