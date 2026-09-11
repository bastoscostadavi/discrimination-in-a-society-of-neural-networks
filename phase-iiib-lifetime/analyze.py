"""Generate a tidy data table, paired summaries, and figures from completed runs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/phase-iiib-mpl')
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
KEYS=['R_muc','R_cw','agenda_cw','sign_cw']
def interval(a):
    a=np.asarray(a);m=a.mean();h=student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0
    return float(m),float(m-h),float(m+h)
def main():
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('N*.json'))]
    tidy=[dict(**{k:r[k] for k in ['N','P','b','f','name','seed']},**v) for r in records for v in r['rows']]
    with (ROOT/'results'/'trajectories.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(tidy[0]));w.writeheader();w.writerows(tidy)
    groups=sorted(set((r['N'],r['P'],r['name']) for r in records))
    summary=[]
    for N,P,name in groups:
        runs=[r for r in records if (r['N'],r['P'],r['name'])==(N,P,name)]
        end=min(r['max_time'] for r in runs)
        for k in KEYS:
            a=np.array([next(v[k] for v in r['rows'] if v['time']==500) for r in runs]);b=np.array([next(v[k] for v in r['rows'] if v['time']==end) for r in runs])
            summary.append(dict(N=N,P=P,name=name,n=len(runs),metric=k,end=end,at500=interval(a),at_end=interval(b),paired_change=interval(b-a)))
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for P in sorted(set(r['P'] for r in records)):
        names=['control','a_reference','f025','b_reference','f075','f090','f100']
        fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,names):
            runs=[r for r in records if r['N']==40 and r['P']==P and r['name']==name]
            if not runs:continue
            times=sorted(set.intersection(*[{v['time'] for v in r['rows'] if v['time']>0} for r in runs]))
            for k,c,ls,label in zip(KEYS,['#b33b30','#2469ad','#168054','#888888'],['-','-','--',':'],['Trust–class','Opinion–class (full)','Opinion–class (agenda)','Stated opinions–class']):
                a=np.array([interval([next(v[k] for v in r['rows'] if v['time']==t) for r in runs]) for t in times])
                ax.plot(times,a[:,0],color=c,ls=ls,label=label);ax.fill_between(times,a[:,1],a[:,2],color=c,alpha=.10)
            ax.set_title(f"b={runs[0]['b']:g}, f={runs[0]['f']:g} ({len(runs)} seeds)")
            ax.axvline(500,color='.65',lw=.8);ax.set_xscale('log');ax.set_ylim(-.08,1.05);ax.grid(alpha=.15)
        axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
        fig.supxlabel('Interactions per ordered pair, Δt');fig.supylabel('Correlation (mean and 95% confidence interval)');fig.suptitle(f'Lifetime of the discriminatory regimes | N=40, K=30, P={P}')
        fig.tight_layout();fig.savefig(ROOT/'figures'/f'trajectories_P{P}.png',dpi=160);fig.savefig(ROOT/'figures'/f'trajectories_P{P}.pdf');plt.close(fig)
    for s in summary:
        if s['metric']=='R_cw':print(s['N'],s['P'],s['name'],s['n'],s['end'],f"{s['at500'][0]:.4f} → {s['at_end'][0]:.4f}; paired Δ {s['paired_change']}")
if __name__=='__main__':main()
