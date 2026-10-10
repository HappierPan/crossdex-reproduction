"""Official four-hand actuator check; scripted actions, no PPO training."""
import os,sys,json,math
from pathlib import Path
os.chdir('/home/siyu/CrossDex/rl');sys.path[:0]=[os.getcwd(),'/home/siyu/CrossDex']
from isaacgym import gymapi
import torch,numpy as np
from omegaconf import OmegaConf
import isaacgymenvs
import tasks_crossdex
from isaacgymenvs.utils.utils import set_seed
from PIL import Image,ImageDraw
OUT=Path('/root/crossdex_validation/four_hand_trial')
base=Path('/root/autodl-tmp/crossdex_experiments/ppo_10000_original_20261002_183343/runs/077_rubiks_cube_2026-10-03_02-33-55/config.json')
raw=json.loads(base.read_text());raw['task']['multi_dex']['hand_names']=['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand'];cfg=OmegaConf.create(raw)
cfg.num_envs=cfg.task.env.numEnvs=16;cfg.test=cfg.task.is_test=cfg.train.params.test=False;cfg.checkpoint='';cfg.task.env.enableCameraSensors=True
set_seed(42,torch_deterministic=cfg.torch_deterministic)
env=isaacgymenvs.make(42,cfg.task_name,16,cfg.sim_device,cfg.rl_device,0,True,False,False,False,cfg)
report={'hands':{},'steps':180,'protocol':'Official environment in training mode; fixed arm targets and smooth scripted eigengrasp actions. This is an actuator check, not a grasp success evaluation.','randomizeRobot':bool(cfg.task.env.randomizeRobot)}
try:
 env.reset();env.step(torch.zeros((16,15),device=env.device));frames={};cams={};props=gymapi.CameraProperties();props.width=640;props.height=480;props.horizontal_fov=65
 armact=torch.zeros((16,15),device=env.device)
 for i,hand in enumerate(env.hand_names):
  model=env.retargeting_models[i];order=env.retarget2isaacs[i];assert len(order)==len(env.hand_dof_names[i]) and len(set(order))==len(order)
  assert [model.robot_joint_names[k] for k in order]==env.hand_dof_names[i]
  indices=env.per_robot_dof_indices[i][env.arm_dof_indices[i]];lo=env.agg_robot_dof_lower_limits[indices];hi=env.agg_robot_dof_upper_limits[indices];q=env.cur_targets[:,indices];armact[i::4,:6]=torch.clamp(2*(q-lo)/(hi-lo)-1,-1,1)
  cams[i]=env.gym.create_camera_sensor(env.envs[i],props);assert cams[i]>=0;frames[i]=[]
  origin=env.gym.get_env_origin(env.envs[i]);center=env.palm_center_pos.reshape(16,3)[i].cpu().numpy();target=gymapi.Vec3(*center.tolist());eye=gymapi.Vec3(*(center+np.array([.55,-.65,.4])).tolist());env.gym.set_camera_location(cams[i],env.envs[i],eye,target)
  report['hands'][hand]={'joint_names':env.hand_dof_names[i],'retarget2isaac':order,'joint_name_mapping_valid':True,'target_max_abs_reconstruction_error_rad':0.,'target_limit_excess_rad':0.,'actual_motion_max_range_rad':0.}
 positions=[]
 for step in range(180):
  action=armact.clone();phase=2*math.pi*step/90;action[:,6:]=.55*math.sin(phase)
  prev=env.prev_targets.clone();reset_before=env.reset_buf.clone().bool()
  obs,reward,done,info=env.step(action);assert torch.isfinite(obs['obs']).all() and torch.isfinite(reward).all();assert torch.isfinite(env.robot_dof_pos).all()
  positions.append(env.robot_dof_pos.detach().cpu().numpy().copy())
  for i,hand in enumerate(env.hand_names):
   idx=env.per_robot_dof_indices[i][env.hand_dof_indices[i]];m=env.retargeting_models[i];coeff=(action[i::4,6:]+1)*.5*(m.max_values-m.min_values)+m.min_values;pred=m.retarget(coeff)[:,env.retarget2isaacs[i]]
   lo=env.agg_robot_dof_lower_limits[idx];hi=env.agg_robot_dof_upper_limits[idx];expected=torch.clamp(.1*torch.clamp(pred,lo,hi)+.9*prev[:,idx],lo,hi);actual=env.cur_targets[:,idx]
   valid=~reset_before[i::4]
   if valid.any():report['hands'][hand]['target_max_abs_reconstruction_error_rad']=max(report['hands'][hand]['target_max_abs_reconstruction_error_rad'],float((expected[valid]-actual[valid]).abs().max()))
   excess=torch.maximum(lo-actual,torch.zeros_like(actual))+torch.maximum(actual-hi,torch.zeros_like(actual));report['hands'][hand]['target_limit_excess_rad']=max(report['hands'][hand]['target_limit_excess_rad'],float(excess.max()))
  if step%3==0:
   env.gym.fetch_results(env.sim,True);env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
   for i,hand in enumerate(env.hand_names):
    rawimage=env.gym.get_camera_image(env.sim,env.envs[i],cams[i],gymapi.IMAGE_COLOR);pic=Image.fromarray(rawimage.reshape(480,640,4)[:,:,:3]);draw=ImageDraw.Draw(pic);draw.rectangle((0,0,640,25),fill='black');draw.text((8,7),f'{hand} | scripted actuator check | step {step+1}',fill='white');frames[i].append(pic)
 pos=np.stack(positions)
 for i,hand in enumerate(env.hand_names):
  idx=env.per_robot_dof_indices[i][env.hand_dof_indices[i]].cpu().numpy();r=report['hands'][hand];r['actual_motion_max_range_rad']=float(np.ptp(pos[:,:,idx],axis=0).max());assert r['target_max_abs_reconstruction_error_rad']<2e-5 and r['target_limit_excess_rad']<1e-6 and r['actual_motion_max_range_rad']>.01
  gif=OUT/(hand+'_actuator.gif');frames[i][0].save(gif,save_all=True,append_images=frames[i][1:],duration=50,loop=0);r['replay']=str(gif);assert np.array(frames[i][-1])[25:].std()>5
 report['passed']=True;(OUT/'environment_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
finally:env.gym.destroy_sim(env.sim)
