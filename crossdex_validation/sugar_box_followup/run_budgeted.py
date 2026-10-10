"""Time-limited sugar-box PPO weight warm start; then fixed-seed evaluation and backup."""
import os,sys,json,time,subprocess,traceback,hashlib,shutil,fcntl
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).parent
OUT=Path('/root/autodl-tmp/crossdex_experiments/sugar_box_warmstart_20261005');OUT.mkdir(parents=True,exist_ok=True)
BASE=Path('/root/crossdex_checkpoints/four_hand_004_sugar_box_10000_20261004')
def now():return datetime.now(timezone.utc).isoformat()
def write(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
if '--train' in sys.argv:
 start=time.monotonic();deadline=start+13800
 repo=Path('/home/siyu/CrossDex');os.chdir(repo/'rl');sys.path[:0]=[str(repo/'rl'),str(repo)]
 from isaacgym import gymapi
 import torch,numpy as np,random
 from omegaconf import OmegaConf
 import run_ppo_multidex
 from algo import ppo
 cfg=OmegaConf.create(json.loads((BASE/'config.json').read_text()))
 cfg.checkpoint=str(BASE/'model_10000.pt');cfg.train.params.max_iterations=10000;cfg.train.params.log_dir=str(OUT/'runs')
 assert cfg.num_envs==8192 and cfg.train.params.save_interval==500 and not cfg.task.env.randomizeRobot
 assert not cfg.test and cfg.task.env.asset.objectAssetFile=='ycb_assets/urdf/004_sugar_box.urdf'
 write(OUT/'launch_manifest.json',{'initialization':'PPO 10000 network weights only; optimizer restarted','source_checkpoint':cfg.checkpoint,'source_sha256':hashlib.sha256(Path(cfg.checkpoint).read_bytes()).hexdigest(),'training_wall_budget_seconds':13800,'max_additional_iterations':10000,'configuration':OmegaConf.to_container(cfg,resolve=True)})
 orig_log=ppo.PPO.log
 class BudgetFinished(Exception):pass
 def budget_log(self,locs,*args,**kwargs):
  orig_log(self,locs,*args,**kwargs)
  it=int(locs['it'])+1
  write(OUT/'training_progress.json',{'additional_iterations_completed':it,'elapsed_seconds':time.monotonic()-start,'budget_seconds':13800,'updated_utc':now()})
  if time.monotonic()>=deadline or it==self.num_learning_iterations:
   path=Path(self.log_dir)/f'model_{it}.pt';self.save(str(path))
   torch.save({'actor_critic':self.actor_critic.state_dict(),'optimizer':self.optimizer.state_dict(),'additional_iterations_completed':it,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'note':'Optimizer and RNG recorded; live simulator state is not included.'},Path(self.log_dir)/'training_state.pt')
   self.writer.flush()
   write(OUT/'training_finished.json',{'checkpoint':str(path),'run':self.log_dir,'additional_iterations_completed':it,'elapsed_seconds':time.monotonic()-start,'finished_utc':now()})
   raise BudgetFinished()
 ppo.PPO.log=budget_log
 try:run_ppo_multidex.main.__wrapped__(cfg)
 except BudgetFinished:print('Training time budget reached; final checkpoint and optimizer saved.',flush=True)
 sys.exit(0)
lock=(P/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status={'stage':'starting','started_utc':now(),'supervisor_pid':os.getpid(),'training_budget_seconds':13800,'object':'004_sugar_box','initialization':'weights_only_from_PPO_10000','hands':['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand']}
def save(**kw):status.update(kw);status['updated_utc']=now();write(OUT/'status.json',status)
def call(cmd,log):
 with (OUT/log).open('w') as f:
  child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);save(child_pid=child.pid,active_log=str(OUT/log))
  while True:
   try:code=child.wait(timeout=30);break
   except subprocess.TimeoutExpired:
    if (OUT/'training_progress.json').exists():status.update(json.loads((OUT/'training_progress.json').read_text()))
    save()
 if code:raise RuntimeError(f'{log} exited {code}')
try:
 save(stage='training');call([sys.executable,'-u',__file__,'--train'],'training.log')
 info=json.loads((OUT/'training_finished.json').read_text());run=Path(info['run']);save(stage='evaluation',**info)
 data=[]
 for seed in [101,202,303,404,505]:
  save(seed=seed)
  result=OUT/f'evaluation_{seed}.json'
  call([sys.executable,'-u','/root/crossdex_validation/evaluate_existing_policy.py','--run',str(run),'--checkpoint',Path(info['checkpoint']).name,'--seed',str(seed),'--envs','256','--output',str(result)],f'evaluation_{seed}.log')
  data.append(json.loads(result.read_text()))
 result={'episodes':1280,'successes':sum(d['native_successes'] for d in data),'per_hand':{}}
 result['success_rate']=result['successes']/1280
 for h in status['hands']:
  n=sum(d['per_hand'][h]['native_successes'] for d in data);result['per_hand'][h]={'episodes':320,'successes':n,'success_rate':n/320}
 baseline=json.loads(Path('/root/autodl-tmp/crossdex_experiments/multi_object_teachers_20261004/004_sugar_box/comparison.json').read_text())[-1]
 write(OUT/'comparison.json',{'baseline_PPO_10000':baseline,'weights_warmstart_followup':result,'training':info})
 dest=Path('/root/crossdex_checkpoints/sugar_box_warmstart_20261005');shutil.copytree(run,dest,dirs_exist_ok=True)
 for name in ['launch_manifest.json','comparison.json','training_finished.json']:shutil.copy2(OUT/name,dest/name)
 write(dest/'sha256_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()})
 save(stage='complete',exit_code=0,backup=str(dest),evaluation=result,finished_utc=now())
except Exception as e:
 save(stage='failed',error=str(e),exit_code=1);traceback.print_exc();sys.exit(1)
