import sys,os,json
from pathlib import Path
repo=Path('/home/siyu/CrossDex');sys.path[:0]=[str(repo/'rl'),str(repo)];os.chdir(repo/'rl')
from isaacgym import gymapi
import torch,isaacgymenvs
import tasks_crossdex
from omegaconf import OmegaConf
from isaacgymenvs.utils.utils import set_seed
base=Path('/root/crossdex_checkpoints/four_hand_ppo_10000_20261003_160542/ppo/config.json')
name=sys.argv[1];cfg=OmegaConf.create(json.loads(base.read_text()))
cfg.num_envs=cfg.task.env.numEnvs=16;cfg.headless=True;cfg.force_render=False
cfg.task.env.asset.objectAssetFile='ycb_assets/urdf/'+name+'.urdf'
set_seed(42,torch_deterministic=cfg.torch_deterministic)
env=isaacgymenvs.make(42,cfg.task_name,16,cfg.sim_device,cfg.rl_device,0,True,False,False,False,cfg)
try:
 obs=env.reset()
 for i in range(120):
  obs,rew,done,info=env.step(torch.zeros((16,env.num_acts),device=env.device))
  assert torch.isfinite(obs['obs']).all() and torch.isfinite(rew).all()
 result={'object':name,'hands':list(env.hand_names),'envs':16,'steps':120,'finite':True,'object_names':env.object_names,'observation_dim':env.observation_space.shape[0],'actions':env.num_acts}
 Path('/root/crossdex_validation/multi_object_teachers/'+name+'_preflight.json').write_text(json.dumps(result,indent=2))
finally:env.gym.destroy_sim(env.sim)
