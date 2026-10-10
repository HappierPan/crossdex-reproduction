import os,sys,json,time,re,subprocess,traceback,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/crossdex_validation/single_object_dagger')
OUT=ROOT/'trial_20261004';OUT.mkdir(parents=True,exist_ok=True)
TEACHER=Path('/root/crossdex_checkpoints/four_hand_ppo_10000_20261003_160542/ppo')
REPO=Path('/home/siyu/CrossDex');MODE=sys.argv[1] if len(sys.argv)>1 else 'supervise'
def write(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
if MODE in ['train','eval']:
 os.chdir(REPO/'rl');sys.path[:0]=[str(REPO/'rl'),str(REPO)]
 from isaacgym import gymapi
 import torch,yaml,isaacgymenvs
 from omegaconf import OmegaConf
 from isaacgymenvs.utils.utils import set_seed
 from run_dagger_multidex import build_runner
 raw=json.loads((TEACHER/'config.json').read_text())
 train=yaml.safe_load((REPO/'rl/tasks_crossdex/train/MultiDexGraspDAGGER.yaml').read_text())
 raw['train']=train;raw['expert']={'077_rubiks_cube':{'path':str(TEACHER),'ckpt':'model_10000.pt'}}
 raw['test_expert']=False;raw['test']=False;raw['task']['is_test']=False
 raw['checkpoint']='';raw['num_envs']=raw['task']['env']['numEnvs']=1024
 raw['headless']=True;raw['force_render']=False;raw['capture_video']=False
 raw['task']['env']['enablePointCloud']=True
 raw['task']['env']['studentObservationType']='armdof+keypts+lastact+objpcl'
 raw['task']['env']['multiTask']=False
 raw['train']['params'].update(max_iterations=1000,save_interval=200,nminibatches=8,log_dir=str(OUT/'runs'))
 cfg=OmegaConf.create(raw)
 if MODE=='eval':
  run=list((OUT/'runs').glob('*/model_1000.pt'))[0].parent
  cfg=OmegaConf.create(json.loads((run/'config.json').read_text()))
  cfg.num_envs=cfg.task.env.numEnvs=256;cfg.seed=101
  cfg.test=cfg.task.is_test=cfg.train.params.test=True
  cfg.checkpoint=str(run/'model_1000.pt');cfg.test_expert=cfg.train.params.test_expert=False
 set_seed(cfg.seed,torch_deterministic=cfg.torch_deterministic)
 env=isaacgymenvs.make(cfg.seed,cfg.task_name,cfg.num_envs,cfg.sim_device,cfg.rl_device,0,True,False,False,False,cfg)
 runner=build_runner(cfg,env)
 if MODE=='train':
  write('launch_manifest.json',{'teacher':str(TEACHER/'model_10000.pt'),'teacher_sha256':hashlib.sha256((TEACHER/'model_10000.pt').read_bytes()).hexdigest(),'envs':1024,'iterations':1000,'pointcloud_source':'official object mesh point clouds transformed to world coordinates','config':OmegaConf.to_container(cfg,resolve=True)})
  runner.run()
 else:
  env.random_time=False;obs=env.reset();active=torch.ones(256,dtype=torch.bool,device=env.device);success=torch.zeros_like(active);length=torch.zeros(256,dtype=torch.long,device=env.device)
  with torch.no_grad():
   for step in range(env.max_episode_length+5):
    assert torch.isfinite(obs['student_obs']).all()
    act=runner.actor_critic.act_inference(obs['student_obs']);assert torch.isfinite(act).all()
    obs,rew,done,info=env.step(act);assert torch.isfinite(rew).all()
    ended=active&done.bool();success|=ended&(env.current_successes>0);length+=active.long();active&=~ended
    if not active.any():break
  assert not active.any()
  hands=list(env.hand_names)
  write('evaluation.json',{'checkpoint':cfg.checkpoint,'seed':101,'protocol':'first completed episode per environment; deterministic student actions; official forced lift test','episodes':256,'successes':int(success.sum()),'success_rate':float(success.float().mean()),'per_hand':{h:{'episodes':64,'successes':int(success[i::4].sum()),'success_rate':float(success[i::4].float().mean())} for i,h in enumerate(hands)}})
 env.gym.destroy_sim(env.sim)
else:
 status={'stage':'starting','started_utc':now(),'supervisor_pid':os.getpid(),'iterations':1000,'envs':1024}
 try:
  for mode in ['train','eval']:
   status.update(stage=mode,updated_utc=now());write('status.json',status)
   with (OUT/(mode+'.log')).open('w') as log:
    child=subprocess.Popen([sys.executable,'-u',__file__,mode],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    status['child_pid']=child.pid;write('status.json',status)
    while child.poll() is None:
     time.sleep(15)
     txt=(OUT/(mode+'.log')).read_text(errors='replace')[-25000:]
     matches=re.findall(r'Learning iteration\s+(\d+)/1000',txt)
     if matches:status['last_logged_iteration']=int(matches[-1])
     status['updated_utc']=now();write('status.json',status)
    if child.returncode:raise RuntimeError(f'{mode} exited {child.returncode}; see {mode}.log')
  status.update(stage='complete',finished_utc=now(),exit_code=0);write('status.json',status)
 except Exception as e:
  status.update(stage='failed',error=str(e),exit_code=1);write('status.json',status);traceback.print_exc();sys.exit(1)
