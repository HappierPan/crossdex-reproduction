"""Local interactive official Isaac Gym policy replay. No training or project file edits."""
import argparse,json,os,sys,hashlib
from pathlib import Path
from datetime import datetime
P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent/'CrossDex')
P.add_argument('--hands',nargs='+',choices=['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand'],default=['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand'])
P.add_argument('--envs',type=int,default=None)
P.add_argument('--seed',type=int,default=101)
P.add_argument('--steps',type=int,default=0,help='0: replay until window closes')
P.add_argument('--headless-check',action='store_true',help='Finite smoke check without GUI (default120 steps)')
a=P.parse_args();bundle=Path(__file__).resolve().parent;repo=a.repo.expanduser().resolve()
assert (repo/'rl/tasks_crossdex/multi_dex_grasp.py').exists(), f'CrossDex project not found: {repo}'
if not a.headless_check and not os.environ.get('DISPLAY'):
 raise SystemExit('No DISPLAY. Run this in a terminal on your local graphical desktop, using the existing Isaac Gym Python environment.')
manifest=json.loads((bundle/'manifest.json').read_text())
for name,sha in manifest['files_sha256'].items():assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==sha, 'Bundle file checksum mismatch: '+name
sys.path[:0]=[str(repo/'rl'),str(repo)];os.chdir(repo/'rl')
try:
 from isaacgym import gymapi
except ModuleNotFoundError:
 raise SystemExit('Isaac Gym is missing from this Python. Use your existing Isaac Gym environment: CROSSDEX_PYTHON=/path/to/its/bin/python bash run.sh')
import torch,yaml
from omegaconf import OmegaConf
import isaacgymenvs
from isaacgymenvs.utils.utils import set_seed
from run_dagger_multidex import build_runner
from retargeting import retargeting_nn_utils as mappings
# Preserve official retargeting/PCA methods; load their inputs from the portable bundle.
# This allows local replay without overwriting local mapping files.
def bundled_init(self,dataset='grab',robot_name=None,add_random_dataset=False,retargeting_type='dexpilot',position_baseline=False,device='cuda',n_eigengrasps=9):
 if not isinstance(robot_name,str):robot_name=mappings.ROBOT_NAME_MAP[robot_name]
 assert dataset=='grab' and not add_random_dataset and not position_baseline and n_eigengrasps==9 and retargeting_type=='dexpilot'
 stem='retargeting_nn_'+robot_name+'_grab_dexpilot'
 with (bundle/'models'/(stem+'.yaml')).open() as f:c=yaml.safe_load(f)
 self.position_baseline=False;self.robot_dim=c['robot_dim'];self.robot_joint_names=c['robot_joint_names'];self.robot_name=c['robot_name'];self.robot_urdf_path=c['robot_urdf_path'];self.device=torch.device(device)
 self.retargeting_nn=mappings.RetargetingNN(robot_dim=self.robot_dim,mano_dim=45).to(self.device)
 self.retargeting_nn.load_state_dict(torch.load(bundle/'models'/(stem+'.pth'),map_location=self.device));self.retargeting_nn.eval()
 values=mappings.load_pca_data(bundle/'pca_9_grab.pkl')
 for name,value in zip(['principal_vectors','min_values','max_values','D_mean','D_std'],values):setattr(self,name,torch.as_tensor(value).to(self.device))
mappings.EigenRetargetModel.__init__=bundled_init
cfg=OmegaConf.create(json.loads((bundle/'config.json').read_text()));n=a.envs or len(a.hands)
assert n>0 and n%len(a.hands)==0
cfg.task.multi_dex.hand_names=a.hands;cfg.num_envs=cfg.task.env.numEnvs=n
cfg.test=cfg.task.is_test=cfg.train.params.test=True;cfg.test_expert=cfg.train.params.test_expert=False;cfg.checkpoint=str(bundle/'model_1000.pt')
cfg.headless=a.headless_check;cfg.capture_video=False;cfg.force_render=False;cfg.task.env.enableCameraSensors=False;cfg.seed=a.seed
set_seed(a.seed,torch_deterministic=cfg.torch_deterministic)
env=isaacgymenvs.make(a.seed,cfg.task_name,n,cfg.sim_device,cfg.rl_device,0,a.headless_check,False,False,False,cfg)
try:
 env.record_frames_dir=str(bundle/'recordings'/datetime.now().strftime('%Y%m%d_%H%M%S'))
 if not a.headless_check:
  assert env.viewer is not None
  env.gym.viewer_camera_look_at(env.viewer,None,gymapi.Vec3(3,-4,2.7),gymapi.Vec3(.7,.7,.5))
 runner=build_runner(cfg,env);env.random_time=False;obs=env.reset()['student_obs'];limit=a.steps or (120 if a.headless_check else 0);step=0
 print('Hands by environment index:',list(enumerate(a.hands)),flush=True)
 print('DAgger student 1000 iterations; mesh-derived object point cloud; inference only. R: start/stop PNG frame recording. Esc: quit. Record directory:',env.record_frames_dir,flush=True)
 with torch.no_grad():
  while limit==0 or step<limit:
   action=runner.actor_critic.act_inference(obs);assert torch.isfinite(action).all();obs,reward,done,info=env.step(action);obs=obs['student_obs'];assert torch.isfinite(obs).all() and torch.isfinite(reward).all();step+=1
 print('Replay check completed:',step,'steps',flush=True)
finally:
 if env.viewer is not None:env.gym.destroy_viewer(env.viewer)
 env.gym.destroy_sim(env.sim)
