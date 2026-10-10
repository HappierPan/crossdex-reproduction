import os,sys,json,subprocess,traceback,hashlib,shutil,fcntl
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).parent;OUT=Path('/root/autodl-tmp/crossdex_experiments/multi_object_teachers_20261004');OUT.mkdir(parents=True,exist_ok=True)
lock=(P/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
objects=['004_sugar_box','005_tomato_soup_can'];state={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'objects':objects,'iterations_per_object':10000,'envs':8192,'completed':[],'automatic_dagger':False}
def save(**kw):
 state.update(kw);state['updated_utc']=datetime.now(timezone.utc).isoformat();t=P/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(P/'status.json')
experts={'077_rubiks_cube':{'path':'/root/crossdex_checkpoints/four_hand_ppo_10000_20261003_160542/ppo','ckpt':'model_10000.pt'}}
try:
 for obj in objects:
  check=json.loads((P/(obj+'_preflight.json')).read_text());assert check['finite'] and check['object_names']==[obj]
  save(stage='training_and_evaluation',active_object=obj,active_output=str(OUT/obj))
  with (OUT/(obj+'_supervisor.log')).open('w') as log:
   child=subprocess.Popen([sys.executable,'-u',str(P/'train_teacher.py'),'--object',obj,'--output',str(OUT/obj)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   save(child_pid=child.pid)
   code=child.wait()
  if code:raise RuntimeError(f'{obj} failed ({code}); see {OUT/obj}')
  run=next((OUT/obj/'runs').glob('*/model_10000.pt')).parent
  dest=Path('/root/crossdex_checkpoints')/('four_hand_'+obj+'_10000_20261004');shutil.copytree(run,dest,dirs_exist_ok=True)
  sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir() if f.is_file()}
  (dest/'sha256_manifest.json').write_text(json.dumps(sha,indent=2))
  experts[obj]={'path':str(dest),'ckpt':'model_10000.pt'}
  state['completed'].append(obj);save(last_backup=str(dest))
  import yaml
  (P/'expert_candidates.yaml').write_text(yaml.safe_dump(experts,sort_keys=False))
 save(stage='complete',note='Teacher evaluations require review before multi-object DAgger launch.')
except Exception as e:
 save(stage='failed',error=str(e));traceback.print_exc();sys.exit(1)
