"""Exact agenda-coordinate dynamics; no imports outside this folder."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/phase-iiib-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, ctypes, json, time, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from ednna.society import SocietyBatch
from ednna.order_params import correlations
ROOT=Path(__file__).resolve().parent
LIB=ROOT/'kernel.so'
def build():
    subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'kernel.cpp'),'-o',str(LIB)],check=True)
def kernel():
    lib=ctypes.CDLL(str(LIB)); fn=lib.evolve
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    ints=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int]*3+[ctypes.c_long]+[arr]*6+[ints]*3+[ctypes.c_double]*2+[ints]
    return fn
class Run:
    def __init__(self,N,P,b,f,seed):
        self.s=SocietyBatch(N,30,P,d=b,f_d=f,seed=seed)
        s=self.s; self.Q=np.linalg.svd(s.X[:,0,:],full_matrices=False)[2].T
        self.w=np.ascontiguousarray(s.w[:,0]@self.Q)
        self.C=np.ascontiguousarray(np.tile(np.eye(self.Q.shape[1]),(N,1,1)))
        self.null=s.w[:,0]-self.w@self.Q.T
        self.mu=np.ascontiguousarray(s.mu[:,:,0]).copy();self.V=np.ascontiguousarray(s.V[:,:,0]).copy()
        self.X=np.ascontiguousarray(s.X[:,0]@self.Q);self.D=np.ascontiguousarray(s.D[:,:,0])
        self.counts=np.zeros(3,dtype=np.int64);self.t=0
        self.fn=kernel()
    def advance(self,target):
        s=self.s; goal=round(target*s.N*(s.N-1))
        while self.t<goal:
            n=min(100000,goal-self.t)
            r=s.rng.integers(s.N,size=n);e=s.rng.integers(s.N-1,size=n);e+=e>=r;p=s.rng.integers(s.P,size=n)
            self.fn(s.N,self.w.shape[1],s.P,n,self.w,self.C,self.mu,self.V,self.X,self.D,r,e,p,s.z_floor,s.v_floor,self.counts)
            self.t+=n
    def measure(self):
        s=self.s;full=self.w@self.Q.T+self.null
        s.w[:,0]=full;s.mu[:,:,0]=self.mu;s.V[:,:,0]=self.V
        out={k:float(v[0]) for k,v in correlations(s).items()}
        def cw(w,mask=None):
            mask=np.ones(s.N,dtype=bool) if mask is None else mask
            w=w[mask];kap=s.kappa[mask];n=len(w)
            if n<2:return float('nan')
            u=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-300)
            return float((np.sum((kap@u)**2)-n)/(n*(n-1)))
        out['agenda_cw']=cw(self.w)
        out['biased_cw']=cw(self.w,s.discriminates[:,0]);out['unbiased_cw']=cw(self.w,~s.discriminates[:,0])
        sig=np.where(self.w@self.X.T>=0,1.,-1.)
        out['sign_cw']=float(np.mean(((s.kappa@sig)**2-s.N)/(s.N*(s.N-1))))
        out['accessible_norm_fraction']=float(np.mean(np.sum(self.w**2,axis=1)/np.sum(full**2,axis=1)))
        out['agenda_C']=float(np.trace(self.C,axis1=1,axis2=2).mean()/self.w.shape[1])
        off=~np.eye(s.N,dtype=bool)
        out['mean_V']=float(self.V[off].mean());out['min_V']=float(self.V[off].min())
        out['min_C_eigenvalue']=float(np.linalg.eigvalsh(self.C).min())
        out.update(dict(zip(['psd_clips','z_floor_hits','v_floor_hits'],map(int,self.counts))))
        out['realized_f']=float(s.discriminates.mean())
        return out
    def save(self,path):
        np.savez_compressed(path,w=self.w,C=self.C,mu=self.mu,V=self.V,X=self.X,D=self.D,Q=self.Q,null=self.null,counts=self.counts,t=self.t,rng=json.dumps(self.s.rng.bit_generator.state))
POINTS=[('control',0.,0.),('a_reference',.5,1.),('f025',1.,.25),('b_reference',1.,.5),('f075',1.,.75),('f090',1.,.9),('f100',1.,1.)]
TIMES=[0,125,500,1000,2500,5000,10000,25000,50000,100000]
def job(spec):
    N,P,name,b,f,seed,maximum=spec; start=time.time()
    tag=f'N{N}_P{P}_{name}_s{seed}'
    path=ROOT/'results'/f'{tag}.json'
    if path.exists() and json.loads(path.read_text())['max_time']>=maximum:return tag+' cached'
    run=Run(N,P,b,f,seed); rows=[]
    for t in [x for x in TIMES if x<=maximum]:
        run.advance(t);rows.append(dict(time=t,**run.measure()))
    run.save(ROOT/'results'/f'{tag}.npz')
    result=dict(N=N,P=P,b=b,f=f,seed=seed,name=name,max_time=maximum,seconds=time.time()-start,rows=rows)
    path.write_text(json.dumps(result,indent=2));return f'{tag}: {result["seconds"]:.1f}s '+str({k:round(rows[-1][k],3) for k in ['R_muc','R_cw','agenda_cw']})
def validate():
    fn=kernel(); reports=[]
    for P in [5,100]:
        run=Run(12,P,1,.5,27);s=run.s;n=5000
        r=s.rng.integers(s.N,size=n);e=s.rng.integers(s.N-1,size=n);e+=e>=r;p=s.rng.integers(P,size=n)
        fn(s.N,run.w.shape[1],P,n,run.w,run.C,run.mu,run.V,run.X,run.D,r,e,p,s.z_floor,s.v_floor,run.counts)
        for rr,ee,pp in zip(r,e,p):s._interact(int(rr),int(ee),s.X[pp])
        errors={'w':float(np.max(abs(s.w[:,0]-(run.w@run.Q.T+run.null)))),'C':float(np.max(abs(s.C[:,0]-(np.einsum('ki,nij,lj->nkl',run.Q,run.C,run.Q)+np.eye(30)-run.Q@run.Q.T)))),'mu':float(np.max(abs(s.mu[:,:,0]-run.mu))),'V':float(np.max(abs(s.V[:,:,0]-run.V)))}
        assert max(errors.values())<1e-9,errors
        reports.append(dict(P=P,interactions=n,max_absolute_error=errors))
    (ROOT/'results'/'validation.json').write_text(json.dumps(reports,indent=2));print(reports,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--validate',action='store_true');ap.add_argument('--max-time',type=int,choices=TIMES[1:],default=10000);ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--issues',type=int,nargs='+',default=[5,100]);ap.add_argument('--points',nargs='+',default=[x[0] for x in POINTS]);ap.add_argument('--agents',type=int,default=40);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    build()
    if args.validate:validate()
    else:
        specs=[(args.agents,P,name,b,f,202609110+seed,args.max_time) for P in args.issues for name,b,f in POINTS if name in args.points for seed in range(args.seeds)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(job,spec) for spec in specs]):print(fut.result(),flush=True)
