import os,sys,json,hashlib
from pathlib import Path
os.chdir('/home/siyu/CrossDex/retargeting');sys.path.insert(0,os.getcwd())
import numpy as np,torch
from retargeting_nn_utils import EigenRetargetModel
from dex_retargeting.robot_wrapper import RobotWrapper
from sklearn.neighbors import NearestNeighbors
out=Path('/root/crossdex_validation/mapping_origin');e=out/'crossdex_mapping_evidence';meta=json.loads((e/'metadata.json').read_text());pca=np.load(e/'pca.npz');trace=np.load('/root/crossdex_validation/trajectory_diagnostics/seed_101.npz');rng=np.random.RandomState(20261003)
def stat(x):
 a=np.asarray(x);return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(a.max())}
def predict(model,x):
 values=[]
 with torch.no_grad():
  for start in range(0,len(x),512):values.append(model.retarget_from_pose45(torch.tensor(x[start:start+512],dtype=torch.float32,device='cuda')).cpu().numpy())
 return np.concatenate(values)
result={}
for i,hand in enumerate(['allegro_hand','shadow_hand']):
 d=meta[hand];data=np.load(e/(hand+'_samples.npz'));model=EigenRetargetModel(robot_name=hand,device='cuda')
 assert hashlib.sha256(Path('/home/siyu/CrossDex/retargeting/models/retargeting_nn_'+hand+'_grab_dexpilot.pth').read_bytes()).hexdigest()==d['weights_sha256']
 assert model.robot_joint_names==d['joint_names'] and not np.intersect1d(data['val_indices'],data['train_indices']).size
 for name in ['eigen_vectors','D_mean','D_std','min_values','max_values']:assert np.array_equal(pca[name],getattr(model,{'eigen_vectors':'principal_vectors'}.get(name,name)).cpu().numpy())
 robot=RobotWrapper(model.robot_urdf_path);ids=[robot.get_joint_index(n) for n in model.robot_joint_names];limits=robot.joint_limits[ids];tips=['link_15.0_tip','link_3.0_tip','link_7.0_tip','link_11.0_tip'] if i==0 else ['thtip','fftip','mftip','rftip','lftip'];tipids=[robot.get_link_index(n) for n in tips];wrist=robot.get_link_index('wrist' if i==0 else 'ee_link')
 def pos(qhand):
  q=robot.q0.copy();q[ids]=qhand;robot.compute_forward_kinematics(q);inv=robot.get_link_pose_inv(wrist);return np.array([(inv@robot.get_link_pose(j))[:3,3] for j in tipids])
 stats={}
 for split in ['train','val']:
  x=data[split+'_pose45'];y=data[split+'_robot_joint_pos'];pred=predict(model,x);err=pred-y;clipped=np.clip(pred,limits[:,0],limits[:,1]);sample=rng.choice(len(x),512,replace=False);tiperrors=[]
  for j in sample:tiperrors.extend(np.linalg.norm(pos(clipped[j])-pos(y[j]),axis=1))
  stats[split]={'samples':len(x),'mse_rad2':float(np.mean(err.astype(np.float64)**2)),'rmse_rad':float(np.sqrt(np.mean(err.astype(np.float64)**2))),'mae_rad':float(abs(err).mean()),'joint_abs_error_rad':stat(abs(err)),'sampled_fk_tip_error_m':stat(tiperrors),'raw_joint_clamp_fraction':float(((pred<limits[:,0]-1e-5)|(pred>limits[:,1]+1e-5)).mean()),'label_joint_limit_excess_fraction':float(((y<limits[:,0]-.002)|(y>limits[:,1]+.002)).mean())}
  np.savez_compressed(out/(hand+'_'+split+'_predictions.npz'),prediction=pred,labels=y)
 valid=trace['active'][:,i::2].copy();valid[0]=False;valid[1:]&=trace['lift_stage'][:-1,i::2]==0
 norm=np.clip(trace['actions'][:,:,6:15][:,i::2][valid],-1,1);coeff=(norm+1)*.5*(pca['max_values']-pca['min_values'])+pca['min_values'];policy=coeff@pca['eigen_vectors']*pca['D_std']+pca['D_mean'];policy=policy[rng.choice(len(policy),512,replace=False)]
 train=data['train_pose45'];val=data['val_pose45'];trainpc=((train-pca['D_mean'])/pca['D_std'])@pca['eigen_vectors'].T;valpc=((val-pca['D_mean'])/pca['D_std'])@pca['eigen_vectors'].T;policypc=((policy-pca['D_mean'])/pca['D_std'])@pca['eigen_vectors'].T
 cov=np.array(d['pca_coeff_covariance']);mu=np.array(d['pca_coeff_mean']);inv=np.linalg.pinv(cov,hermitian=True)
 def distance(x):z=x-mu;return np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',z,inv,z),0))
 train_dist=distance(trainpc);val_dist=distance(valpc);policy_dist=distance(policypc);thresh=float(np.quantile(val_dist,.99))
 nn=NearestNeighbors(n_neighbors=1,algorithm='brute',n_jobs=4).fit((train-pca['D_mean'])/pca['D_std'])
 vd=nn.kneighbors((val[:512]-pca['D_mean'])/pca['D_std'])[0][:,0];pd=nn.kneighbors((policy-pca['D_mean'])/pca['D_std'])[0][:,0]
 curves=d['loss_curves'];last_train=curves['Loss/Train-epoch'][-1]['value'];best_val=min(v['value'] for v in curves['Loss/Val']);last_val=curves['Loss/Val'][-1]['value'];first50=min(v['value'] for v in curves['Loss/Val'][-50:]);early50=min(v['value'] for v in curves['Loss/Val'][-100:-50])
 result[hand]={'weights_match_export':True,'pca_matches_export':True,'exported_splits_disjoint':True,'fits':stats,'loss_summary':{'last_train_mse_rad2':last_train,'best_val_mse_rad2':best_val,'last_val_mse_rad2':last_val,'relative_best_val_improvement_recent_50_epochs':float((early50-first50)/early50)},'domain':{'pca_mahalanobis_radius_train':stat(train_dist),'pca_mahalanobis_radius_val':stat(val_dist),'pca_mahalanobis_radius_policy':stat(policy_dist),'val_p99_radius':thresh,'policy_fraction_above_val_p99_radius':float((policy_dist>thresh).mean()),'standardized_pose_nearest_training_sample_distance_val':stat(vd),'standardized_pose_nearest_training_sample_distance_policy':stat(pd),'coverage_limit':'Mahalanobis and nearest-neighbor distances are diagnostics; original split is frame-random, not a subject/sequence-heldout test.'}}
 (out/'uploaded_evidence_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(hand,json.dumps(result[hand]),flush=True)
print('Uploaded evidence analysis complete',flush=True)
