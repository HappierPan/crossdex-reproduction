"""Evaluate first complete episode in each environment; no learning or rendering."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--checkpoint',required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--envs',type=int,default=16)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
repo=Path('/home/siyu/CrossDex'); sys.path[:0]=[str(repo/'rl'),str(repo)];os.chdir(repo/'rl')
from isaacgym import gymapi
import torch
from omegaconf import OmegaConf
import isaacgymenvs
from isaacgymenvs.utils.utils import set_seed
from run_dagger_multidex import build_runner
cfg=OmegaConf.create(json.loads((a.run/'config.json').read_text()))
hands=list(cfg.task.multi_dex.hand_names)
assert a.envs>0 and a.envs%len(hands)==0
cfg.test=cfg.task.is_test=cfg.train.params.test=True
cfg.test_expert=cfg.train.params.test_expert=False
cfg.num_envs=cfg.task.env.numEnvs=a.envs
cfg.headless=True;cfg.capture_video=False;cfg.force_render=False;cfg.task.env.enableCameraSensors=True
cfg.seed=a.seed;cfg.checkpoint=str(a.run/a.checkpoint)
set_seed(a.seed,torch_deterministic=cfg.torch_deterministic)
env=isaacgymenvs.make(a.seed,cfg.task_name,a.envs,cfg.sim_device,cfg.rl_device,0,True,False,False,False,cfg)
from PIL import Image, ImageDraw
selection=json.loads(Path('/root/crossdex_validation/single_object_dagger/expanded/replay/selection.json').read_text())['selected_samples']
cameras={};frames={}
props=gymapi.CameraProperties();props.width=960;props.height=720;props.horizontal_fov=75
for sample in selection:
 j=sample['env_id'];handle=env.gym.create_camera_sensor(env.envs[j],props)
 assert handle>=0, 'Camera creation failed'
 env.gym.set_camera_location(handle,env.envs[j],gymapi.Vec3(0.85,-1.1,1.1),gymapi.Vec3(-0.25,0.0,0.45))
 cameras[j]=handle;frames[j]=[]
print('Created selected replay cameras',flush=True)
try:
 runner=build_runner(cfg,env);env.random_time=False
 obs=env.reset()['student_obs'];device=env.device
 active=torch.ones(a.envs,device=device,dtype=torch.bool)
 streak=torch.zeros(a.envs,device=device,dtype=torch.long)
 strict30=torch.zeros_like(active);strict60=torch.zeros_like(active)
 native=torch.zeros_like(active);returns=torch.zeros(a.envs,device=device)
 heights=torch.full((a.envs,),-float('inf'),device=device)
 lengths=torch.zeros(a.envs,device=device,dtype=torch.long)
 with torch.no_grad():
  for step in range(env.max_episode_length+5):
   action=runner.actor_critic.act_inference(obs)
   assert torch.isfinite(action).all()
   nxt,reward,done,info=env.step(action);obs=nxt['student_obs']
   assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
   obj=env.object_pos
   palm=(env.palm_center_pos.reshape(a.envs,3)-obj).norm(dim=-1)
   tip=torch.empty(a.envs,device=device)
   for i in range(env.num_robots):
    offset=env.agg_num_fingers[i];count=env.num_fingers[i]
    tip[i::env.num_robots]=(env.fingertip_pos[:,offset:offset+count]-obj[i::env.num_robots,None,:]).norm(dim=-1).mean(dim=-1)
   good=((obj[:,2]-env.goal_height).abs()<=env.success_tolerance)&((palm<=0.15)|(tip<=0.12))
   streak=torch.where(active&good,streak+1,torch.where(active,torch.zeros_like(streak),streak))
   strict30|=active&(streak>=30);strict60|=active&(streak>=60)
   returns+=torch.where(active,reward,torch.zeros_like(reward));lengths+=active.long()
   heights=torch.where(active,torch.maximum(heights,obj[:,2]),heights)
   ended=active&done.bool();native|=ended&(env.current_successes>0)
   if step%3==0 or any(bool(ended[j]) for j in cameras):
    env.gym.fetch_results(env.sim,True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    for sample in selection:
     j=sample['env_id']
     if not bool(active[j]):continue
     raw=env.gym.get_camera_image(env.sim,env.envs[j],cameras[j],gymapi.IMAGE_COLOR)
     pic=Image.fromarray(raw.reshape(props.height,props.width,4)[:,:,:3])
     draw=ImageDraw.Draw(pic);draw.rectangle((0,0,960,26),fill='black')
     draw.text((8,7),f"{sample['hand']} | env {j} | step {step+1} | DAgger 1000 | official test_forced_lift",fill='white')
     frames[j].append((step+1,pic))
   active&=~ended
   if not active.any():break
 assert not active.any(), 'Incomplete episodes; do not report as full evaluation.'
 result={'checkpoint':cfg.checkpoint,'seed':a.seed,'episodes':a.envs,
 'native_successes':native.sum().item(),'strict30_successes':strict30.sum().item(),'strict60_successes':strict60.sum().item(),
 'mean_return':returns.mean().item(),'mean_length':lengths.float().mean().item(),
 'per_episode_success':native.cpu().tolist(),'per_episode_return':returns.cpu().tolist(),'per_episode_length':lengths.cpu().tolist(),'max_object_height':heights.max().item(),'per_episode_max_height':heights.tolist(),
 'protocol':'First completed episode per env, deterministic actions, fixed seed. Native termination uses runtime test_forced_lift setting; strict30/60 are auxiliary streak counts before native termination, not independent evaluation protocols.',
 'native_test_forced_lift':bool(env.test_forced_lift),
 'native_n_lift_steps':int(env.n_lift_steps) if env.test_forced_lift else None}
 import numpy as np
 a.output.parent.mkdir(parents=True,exist_ok=True)
 video_results=[]
 for sample in selection:
  j=sample['env_id'];seq=frames[j];assert len(seq)>=2
  stem=sample['hand']+'_'+sample['outcome']
  duration=[max(10,round((seq[k+1][0]-seq[k][0])*float(env.dt)*1000)) for k in range(len(seq)-1)]+[50]
  gif=a.output.parent/(stem+'.gif')
  seq[0][1].save(gif,save_all=True,append_images=[p for _,p in seq[1:]],duration=duration,loop=0)
  last=np.array(seq[-1][1]);assert float(last[26:].std())>5, 'Blank camera frame'
  seq[-1][1].save(a.output.parent/(stem+'_last_frame.png'))
  video_results.append(dict(sample,gif=str(gif),frames=len(seq),observed_success=bool(native[j]),actual_length=int(lengths[j]),matches_expected_outcome=bool(native[j])==(sample['outcome']=='success'),matches_expected_length=int(lengths[j])==sample['length']))
 expected=json.loads(Path('/root/crossdex_validation/single_object_dagger/expanded/student_101.json').read_text())
 result['per_episode_outcomes_match_nonrendered']=native.cpu().tolist()==expected['per_episode_success']
 result['camera_recordings']=video_results
 assert all(v['matches_expected_outcome'] and v['matches_expected_length'] for v in video_results)
 assert result['per_episode_outcomes_match_nonrendered']
 assert list(env.hand_names)==hands
 result['per_hand']={}
 for i,hand in enumerate(hands):
  result['per_hand'][hand]={'episodes':a.envs//len(hands),'native_successes':int(native[i::len(hands)].sum()),'mean_return':float(returns[i::len(hands)].mean()),'strict30_successes':int(strict30[i::len(hands)].sum())}
 result['hand_names']=hands
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:
 env.gym.destroy_sim(env.sim)
