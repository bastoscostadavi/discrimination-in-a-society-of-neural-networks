"""Resume actual late societies; retain baseline checkpoints and results."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from experiment import ROOT,Run,build
OUT=ROOT/'results'/'probes'
def one(spec):
 seed,arm=spec;tag=f'N40_P5_b_reference_s{seed}';meta=json.loads((ROOT/'results'/f'{tag}.json').read_text());a=np.load(ROOT/'results'/f'{tag}.npz')
 run=Run(40,5,1,.5,seed)
 for k in ['w','C','mu','V','X','D','Q','null','counts']:setattr(run,k,a[k].copy())
 run.t=int(a['t']);run.s.rng.bit_generator.state=json.loads(str(a['rng']))
 assert run.t==100000*40*39
 sig0=np.sign(run.w@run.X.T);w0=run.w.copy();start=time.time()
 before=run.measure()
 if arm=='perturb':
  rng=np.random.default_rng(seed+9000);noise=rng.normal(size=run.w.shape);noise/=np.linalg.norm(noise,axis=1,keepdims=True)
  run.w+=.01*np.linalg.norm(run.w,axis=1,keepdims=True)*noise
 elif arm=='reset_C':run.C[:]=np.eye(run.w.shape[1])
 rows=[dict(time=100000,**run.measure())]
 targets=[101000,110000,200000] if arm!='baseline' else [101000,110000,200000,500000,1000000]
 for t in targets:
  run.advance(t);r=run.measure();r['fraction_changed_signs']=float(np.mean(np.sign(run.w@run.X.T)!=sig0));rows.append(dict(time=t,**r))
 run.save(OUT/f'{tag}_{arm}.npz')
 result=dict(seed=seed,arm=arm,before=before,rows=rows,seconds=time.time()-start)
 (OUT/f'{tag}_{arm}.json').write_text(json.dumps(result,indent=2))
 return arm,seed,round(result['seconds'],1),rows[-1]['R_cw'],rows[-1]['fraction_changed_signs']
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arms',nargs='+',default=['baseline','perturb','reset_C']);ap.add_argument('--seeds',type=int,default=12);args=ap.parse_args();OUT.mkdir(exist_ok=True);build()
 with ProcessPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(one,(202609110+i,arm)) for arm in args.arms for i in range(args.seeds)]):print(f.result(),flush=True)
