"""Detached original-config four-hand 10000-iteration PPO validation, followed by fixed-seed evaluation."""
import argparse,json,os,sys,subprocess,fcntl,re,hashlib,traceback,time
from datetime import datetime,timezone
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--output',type=Path,required=True);P.add_argument('--train',action='store_true');a=P.parse_args()
REPO=Path('/home/siyu/CrossDex')
BASE=Path('/root/autodl-tmp/crossdex_experiments/four_hand_short_20261003_143524/runs/077_rubiks_cube_2026-10-03_22-35-38/config.json')
OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,d):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def original_weights():
 expected={'allegro_hand':'4d970e85d3cbdb1c2cd60075404021a3166eb5d4d9c84432abf44efce0dae01c','shadow_hand':'069debac8f11f36667b072c73232be4bdaf4a3fadb9a32833722e6d3fb39d4e0','ability_hand':'79cd7e32cba1807906e84567854e2a7d02d28bc5e4d6c6087415e31ea22d2f45','schunk_svh_hand':'c4fd1b4c244365ba267296a9bd0aa8df52cbbc17cdc044f0a306a86ac227c105'}
 for hand,sha in expected.items():
  path=REPO/'retargeting/models'/('retargeting_nn_'+hand+'_grab_dexpilot.pth');assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,hand
 return expected
if a.train:
 os.chdir(REPO/'rl');sys.path[:0]=[str(REPO/'rl'),str(REPO)]
 from isaacgym import gymapi
 from omegaconf import OmegaConf
 import run_ppo_multidex
 raw=json.loads(BASE.read_text());changed=json.loads(BASE.read_text())
 changed['task']['multi_dex']['hand_names']=['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand']
 changed['train']['params']['max_iterations']=10000;changed['train']['params']['log_dir']=str(OUT/'runs')
 assert changed['checkpoint']=='' and not changed['test'] and not changed['train']['params']['test'] and not changed['task']['is_test']
 assert changed['seed']==42 and changed['num_envs']==8192 and changed['train']['params']['save_interval']==500
 assert changed['task']['multi_dex']['hand_names']==['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand'] and not changed['task']['multi_dex']['add_random_dataset']
 diff=[]
 def differences(x,y,p=''):
  if isinstance(x,dict):
   for k in x: differences(x[k],y[k],p+'.'+k if p else k)
  elif x!=y:diff.append({'path':p,'old':x,'new':y})
 differences(raw,changed)
 assert {d['path'] for d in diff}=={'train.params.max_iterations','train.params.log_dir'}
 atomic(OUT/'launch_manifest.json',{'source_config':str(BASE),'changes':diff,'mapping_weight_sha256':original_weights(),'initialization':'from_scratch','seed':42,'iterations':10000,'started_utc':now(),'evaluation_checkpoints':[1000,5000,10000],'evaluation_seeds':[101,202,303,404,505],'evaluation_envs':256})
 run_ppo_multidex.main.__wrapped__(OmegaConf.create(changed));sys.exit(0)
lock=(OUT.parent/'ppo_four_hand_10000.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status={'stage':'starting','iterations':10000,'envs':8192,'hands':['shadow_hand','allegro_hand','ability_hand','schunk_svh_hand'],'initialization':'from_scratch','supervisor_pid':os.getpid(),'started_utc':now()}
def save(**kw):status.update(kw);status['updated_utc']=now();atomic(OUT/'status.json',status)
def run(cmd,log):
 with (OUT/log).open('w') as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);save(child_pid=child.pid,active_log=str(OUT/log))
  while True:
   try:code=child.wait(timeout=30);break
   except subprocess.TimeoutExpired:
    if log=='training.log':
     with (OUT/log).open('rb') as tail:tail.seek(max(0,(OUT/log).stat().st_size-32768));txt=tail.read().decode(errors='replace')
     matches=re.findall(r'Learning iteration\s+(\d+)/10000',txt)
     if matches:save(last_logged_iteration=int(matches[-1]))
    else:save()
 if code:raise RuntimeError(f'{log}: exit {code}')
try:
 assert json.loads(Path('/root/crossdex_validation/four_hand_trial/environment_check.json').read_text())['passed']
 original_weights();save(stage='training');run([sys.executable,'-u',__file__,'--output',str(OUT),'--train'],'training.log')
 candidates=list((OUT/'runs').glob('*/model_10000.pt'));assert len(candidates)==1
 folder=candidates[0].parent;save(stage='evaluation',run_directory=str(folder),last_logged_iteration=9999)
 rows=[]
 for iteration in [1000,5000,10000]:
  ds=[]
  for seed in [101,202,303,404,505]:
   name=f'evaluation_{iteration}_{seed}';save(checkpoint=iteration,seed=seed)
   run([sys.executable,'-u','/root/crossdex_validation/evaluate_existing_policy.py','--run',str(folder),'--checkpoint',f'model_{iteration}.pt','--seed',str(seed),'--envs','256','--output',str(OUT/(name+'.json'))],name+'.log')
   ds.append(json.loads((OUT/(name+'.json')).read_text()))
  row={'iteration':iteration,'episodes':sum(d['episodes'] for d in ds),'native_successes':sum(d['native_successes'] for d in ds),'per_hand':{}}
  row['success_rate']=row['native_successes']/row['episodes']
  for hand in status['hands']:
   episodes=sum(d['per_hand'][hand]['episodes'] for d in ds);successes=sum(d['per_hand'][hand]['native_successes'] for d in ds);row['per_hand'][hand]={'episodes':episodes,'native_successes':successes,'success_rate':successes/episodes}
  rows.append(row);atomic(OUT/'comparison.json',rows)
 original_weights();save(stage='complete',exit_code=0,finished_utc=now())
except Exception as error:
 save(stage='failed',exit_code=1,error=str(error),finished_utc=now());traceback.print_exc();sys.exit(1)
