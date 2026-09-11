import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/phase-iiib-mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import interval
ROOT=Path(__file__).resolve().parent

def main():
 baselines=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('N*.json'))]
 probes=[json.loads(p.read_text()) for p in sorted((ROOT/'results'/'probes').glob('*.json'))]
 summary={}
 for arm in ['baseline','perturb','reset_C']:
  runs=[r for r in probes if r['arm']==arm]
  if not runs:continue
  d={'n':len(runs),'end':runs[0]['rows'][-1]['time']}
  for k in ['R_cw','R_muc','agenda_cw','biased_cw','unbiased_cw']:
   final=np.array([r['rows'][-1][k] for r in runs]);initial=np.array([r['before'][k] for r in runs])
   d[k]={'end':interval(final),'change_since_100000':interval(final-initial)}
  d['societies_with_changed_signs']=sum(r['rows'][-1]['fraction_changed_signs']>0 for r in runs)
  d['max_fraction_changed_signs']=max(v.get('fraction_changed_signs',0) for r in runs for v in r['rows'])
  if arm=='baseline':
   init=np.array([next(v['R_cw'] for v in next(b for b in baselines if b['N']==40 and b['P']==5 and b['name']=='b_reference' and b['seed']==r['seed'])['rows'] if v['time']==500) for r in runs])
   end=np.array([r['rows'][-1]['R_cw'] for r in runs]);d['R_cw_change_since_500']=interval(end-init);d['R_cw_range_at_end']=[float(end.min()),float(end.max())]
  if arm!='baseline':
   matched=[(r,next((b for b in probes if b['arm']=='baseline' and b['seed']==r['seed']),None)) for r in runs]
   delta=[r['rows'][-1]['R_cw']-next(v['R_cw'] for v in b['rows'] if v['time']==200000) for r,b in matched if b]
   d['paired_R_cw_difference_vs_unmodified_at_200000']=interval(delta)
  summary[arm]=d
 rows=[v for r in baselines+probes for v in r['rows']]
 summary['numerics']={k:max(r[k] for r in rows) for k in ['psd_clips','z_floor_hits','v_floor_hits']}
 summary['numerics']['min_C_eigenvalue']=min(r['min_C_eigenvalue'] for r in rows)
 summary['numerics']['min_V']=min(r['min_V'] for r in rows)
 summary['baseline_societies']=len(baselines);summary['probe_branches']=len(probes)
 (ROOT/'results'/'final_summary.json').write_text(json.dumps(summary,indent=2))
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,2,figsize=(11,7.5),sharex=True)
 def draw(ax,trajectories,key,color,label,ls='-'):
  times=sorted(set.intersection(*[set(v['time'] for v in rows if v['time']>=125) for rows in trajectories]))
  stats=np.array([interval([next(v[key] for v in rows if v['time']==t) for rows in trajectories]) for t in times])
  ax.plot(times,stats[:,0],color=color,label=label,ls=ls,lw=2);ax.fill_between(times,stats[:,1],stats[:,2],color=color,alpha=.15)
 trajectories={}
 for name in ['a_reference','b_reference']:
  rr=[r for r in baselines if r['N']==40 and r['P']==5 and r['name']==name]
  trajectories[name]=[]
  for r in rr:
   rows=r['rows'].copy()
   if name=='b_reference':
    pp=next((p for p in probes if p['seed']==r['seed'] and p['arm']=='baseline'),None)
    if pp:rows+=pp['rows'][1:]
   trajectories[name].append(rows)
 for ax,key,title in zip(axs.flat,['R_muc','R_cw','agenda_cw'],['Trust follows class','Full-space opinion alignment','Agenda-space opinion alignment']):
  for name,c,label in [('a_reference','#2668a0','IIIa reference: b=0.5, f=1'),('b_reference','#bd5032','IIIb reference: b=1, f=0.5')]:draw(ax,trajectories[name],key,c,label)
  ax.set_title(title)
 draw(axs[1,1],trajectories['b_reference'],'biased_cw','#7c4b9c','Biased agents (IIIb)')
 draw(axs[1,1],trajectories['b_reference'],'unbiased_cw','#27866f','Class-blind agents (IIIb)')
 axs[1,1].set_title('Agenda alignment within each subgroup')
 for ax in axs.flat:
  ax.set_xscale('log');ax.axvline(500,color='.55',lw=1,ls=':');ax.set_ylim(-.1,1.07);ax.grid(alpha=.15);ax.legend(fontsize=9,frameon=False,loc='best')
 fig.suptitle('IIIb reference: weak full-space alignment persists for 2,000× longer\nN=40, K=30, P=5; 12 seeds; mean and 95% confidence interval',fontsize=13)
 fig.supxlabel('Interactions per ordered pair, Δt (dotted line: paper’s Δt=500)');fig.supylabel('Class correlation');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'central_result.{ext}',dpi=180)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
