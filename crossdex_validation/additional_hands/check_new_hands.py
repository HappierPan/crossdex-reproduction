import os,sys,json
from pathlib import Path
os.chdir('/home/siyu/CrossDex/retargeting');sys.path.insert(0,os.getcwd())
import numpy as np,torch
from manopth.manolayer import ManoLayer
from dex_retargeting.constants import RobotName,RetargetingType,HandType,get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from retargeting_nn_utils import RetargetingNN
from lxml import etree
out=Path('/root/crossdex_validation/additional_hands');RetargetingConfig.set_default_urdf_dir('/home/siyu/crossdex-deps/dex-retargeting/assets/robots/hands')
a=np.load('/root/crossdex_validation/mapping_origin/crossdex_mapping_evidence/allegro_hand_samples.npz');poses=a['train_pose45'][:128]
mano=ManoLayer(mano_root='../mano-models',use_pca=False,ncomps=45,flat_hand_mean=True);mp=torch.zeros((len(poses),48));mp[:,3:]=torch.tensor(poses)
with torch.no_grad():human=mano(mp)[1].numpy()/1000.
report={}
for hand,enum,asset in [('ability_hand',RobotName.ability,'rm65_ability_right.urdf'),('schunk_svh_hand',RobotName.svh,'rm65_svh_right.urdf')]:
 cfg=RetargetingConfig.load_from_file(get_default_config_path(enum,RetargetingType.dexpilot,HandType.right),override={'add_dummy_free_joint':True});seq=cfg.build();idx=seq.optimizer.target_link_human_indices;q=np.array([seq.retarget(h[idx[1]]-h[idx[0]])[6:] for h in human],dtype=np.float32)
 names=seq.joint_names[6:];robot=seq.optimizer.robot;limits=robot.joint_limits[6:];assert len(names)==q.shape[1] and np.isfinite(q).all()
 assert np.max(np.maximum(limits[:,0]-q,0)+np.maximum(q-limits[:,1],0)) <= 0.00101  # Official optimizer epsilon=1e-3
 urdf=Path('/home/siyu/CrossDex/robot_randomization/urdf_origin')/asset;tree=etree.parse(str(urdf));arm_names=[j.get('name') for j in tree.getroot().findall('joint') if j.get('type')!='fixed' and j.get('name') not in ['arm_joint'+str(i) for i in range(1,7)]]
 assert set(names)==set(arm_names),(names,arm_names)
 model=RetargetingNN(robot_dim=len(names));opt=torch.optim.Adam(model.parameters());x=torch.tensor(poses);y=torch.tensor(q)
 initial=float(((model(x)-y)**2).mean())
 for _ in range(50):opt.zero_grad();loss=((model(x)-y)**2).mean();loss.backward();opt.step()
 final=float(((model(x)-y)**2).mean());assert final<initial
 np.savez_compressed(out/(hand+'_smoke_pairs.npz'),mano_pose45=poses,robot_joint_pos=q)
 report[hand]={'samples':len(q),'joint_names':names,'robot_dim':len(names),'joint_name_set_matches_arm_hand_urdf':True,'arm_hand_joint_order':arm_names,'retarget_to_asset_indices':[arm_names.index(n) for n in names],'max_limit_excess_rad':float(np.max(np.maximum(limits[:,0]-q,0)+np.maximum(q-limits[:,1],0))),'training_smoke_initial_mse':initial,'training_smoke_final_mse':final,'production_weights_written':False}
 print(hand,report[hand],flush=True)
(out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
