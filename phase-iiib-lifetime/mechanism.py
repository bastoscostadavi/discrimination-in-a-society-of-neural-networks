"""Frozen-environment drift diagnostic at late checkpoints, not a proof of global stability."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent

def diagnose(path):
 a=np.load(path);w,C,mu,V,X,D=[a[k] for k in ['w','C','mu','V','X','D']];N,q=w.shape
 biased=np.any(D!=0,axis=1);kap=np.where(np.arange(N)<N//2,1.,-1.)
 sig=np.where(w@X.T>=0,1.,-1.);eta=1-2*ndtr(mu/np.sqrt(1+V));G=kap[:,None]*kap[None,:];off=~np.eye(N,dtype=bool)
 alignment=(sig@sig.T)/X.shape[0]
 norms=np.linalg.norm(w,axis=1)
 out={'file':path.name,'time':float(a['t'])/(N*(N-1)),'biased_norm':float(norms[biased].mean()),'unbiased_norm':float(norms[~biased].mean()) if (~biased).any() else None}
 for label,mask in [('biased',biased),('unbiased',~biased)]:
  if not mask.any():continue
  pair=off&mask[:,None]
  out[label+'_trust_class']=float(np.mean((eta*G)[pair]));out[label+'_trust_opinion_sign']=float(np.mean((eta*alignment)[pair]))
 rows=[]
 for r in np.flatnonzero(biased):
  es=np.arange(N)!=r
  gc=np.sqrt(1+np.einsum('pi,ij,pj->p',X,C[r],X));features=(sig[es,:,None]*X[None,:,:]/gc[None,:,None]).reshape(-1,q)
  shift=np.repeat(D[r,es],len(X));pm=np.repeat(ndtr(mu[r,es]/np.sqrt(1+V[r,es])),len(X))
  def objective(u):
   h=features@u+shift;pw=ndtr(h);z=np.maximum(pw+pm-2*pw*pm,1e-12)
   fw=(1-2*pm)*np.exp(-h*h/2)/np.sqrt(2*np.pi)/z
   return -np.log(z).mean(),-features.T@fw/len(h)
  opt=minimize(objective,w[r],jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':500})
  h=features@opt.x+shift;pw=ndtr(h);z=np.maximum(pw+pm-2*pw*pm,1e-12);fw=(1-2*pm)*np.exp(-h*h/2)/np.sqrt(2*np.pi)/z;fc=-fw*(fw+h)
  hessian=features.T@(fc[:,None]*features)/len(h)
  rows.append({'agent':int(r),'norm':float(norms[r]),'equilibrium_norm':float(np.linalg.norm(opt.x)),'distance_to_conditional_equilibrium':float(np.linalg.norm(opt.x-w[r])),'gradient_norm_now':float(np.linalg.norm(objective(w[r])[1])),'gradient_norm_equilibrium':float(np.linalg.norm(opt.jac)),'max_eigenvalue_log_evidence_hessian':float(np.linalg.eigvalsh(hessian).max())})
 out['biased_agents']=rows
 return out
if __name__=='__main__':
 data=[diagnose(p) for p in sorted((ROOT/'results').glob('N40_P5_b_reference*.npz'))]
 (ROOT/'results'/'mechanism.json').write_text(json.dumps(data,indent=2))
 for k in ['biased_norm','unbiased_norm','biased_trust_class','unbiased_trust_class','biased_trust_opinion_sign','unbiased_trust_opinion_sign']:
  print(k,np.mean([d[k] for d in data]))
 rows=[r for d in data for r in d['biased_agents']]
 for k in ['distance_to_conditional_equilibrium','gradient_norm_now','gradient_norm_equilibrium','max_eigenvalue_log_evidence_hessian']:
  print(k,'mean',np.mean([r[k] for r in rows]),'max',max(r[k] for r in rows))
