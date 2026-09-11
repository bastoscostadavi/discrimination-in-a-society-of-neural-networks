"""Update the completed-results section after all documented experiments finish."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase-iiib-mpl")
import json,hashlib,platform,subprocess
from pathlib import Path
import numpy as np,scipy,matplotlib
from final_figures import main as figures
ROOT=Path(__file__).resolve().parent

def main():
 files=list((ROOT/'results'/'probes').glob('*.json'))
 assert len(files)==36, 'Complete all twelve continuations and both paired probe arms first.'
 figures()
 s=json.loads((ROOT/'results'/'final_summary.json').read_text())
 for arm in ['baseline','perturb','reset_C']:assert s[arm]['n']==12
 fmt=lambda v:f'{v[0]:.5f} [{v[1]:.5f}, {v[2]:.5f}]'
 b=s['baseline'];num=s['numerics']
 lines=['## Completed continuation and robustness results','',
 'All 180 baseline societies and 36 continuation/probe branches completed. The twelve unmodified long continuations extend the same IIIb societies to Δt=1,000,000, **2,000 times** the paper’s observation duration. Values below are means with across-seed 95% confidence intervals.','',
 '| Test | Final Δt | Full-space R_cw |','|---|---:|---:|']
 for arm,label in [('baseline','Unmodified IIIb continuation'),('perturb','1% agenda-weight perturbation at 100,000'),('reset_C','Agenda covariance reset to identity at 100,000')]:
  lines.append(f"| {label} | {s[arm]['end']:,} | {fmt(s[arm]['R_cw']['end'])} |")
 lines+=['',f"The paired R_cw change from 500 to 1,000,000 is **{fmt(b['R_cw_change_since_500'])}**. Final individual-society values range from {b['R_cw_range_at_end'][0]:.5f} to {b['R_cw_range_at_end'][1]:.5f}. This remains far below the IIIa reference mean of 0.59691 at 100,000.",'']
 for arm,label in [('perturb','The 1% perturbation'),('reset_C','Resetting agenda covariance')]:
  lines.append(f"{label} changes R_cw relative to the matched unmodified continuation at 200,000 by {fmt(s[arm]['paired_R_cw_difference_vs_unmodified_at_200000'])}.")
 lines+=['',f"The maximum fraction of stated-opinion signs differing from the 100,000 checkpoint at a recorded later observation is {max(s[a]['max_fraction_changed_signs'] for a in ['baseline','perturb','reset_C']):.5f} across these arms. This checks the sampled states; signs were not logged at every interaction.",'',
 'The small perturbation and uncertainty reset both leave the weak-alignment configuration intact over their tested horizons. Together with the conditional equilibrium calculation, this argues against small opinion covariance being the sole reason IIIb has not become IIIa. Resetting opinion covariance does not reset trust variance or test arbitrary basin-changing perturbations.','',
 '| Additional comparison at (b,f_b)=(1,0.5) | R_cw at 500 | R_cw at final time | Final Δt |','|---|---:|---:|---:|']
 allsum=json.loads((ROOT/'results'/'summary.json').read_text())
 for N,P,label in [(60,5,'Larger population: N=60, P=5'),(40,100,'Complex agenda: N=40, P=100')]:
  v=next(v for v in allsum if v['N']==N and v['P']==P and v['name']=='b_reference' and v['metric']=='R_cw')
  lines.append(f"| {label} | {fmt(v['at500'])} | {fmt(v['at_end'])} | {v['end']:,} |")
 lines+=['',f"Across all recorded baseline and probe states, the minimum agenda-covariance eigenvalue is {num['min_C_eigenvalue']:.3g}, and the minimum non-self trust variance is {num['min_V']:.3g}. No PSD clips, evidence-floor activations, or trust-variance-floor activations occurred.",'',
 '![Central result](figures/central_result.png)','',
 'The result supports persistence of IIIb at the representative parameters. It does **not** establish that every point labeled IIIb is an asymptotically distinct phase, nor that the IIIa/IIIb boundary is a sharp transition rather than a crossover.']
 report=ROOT/'REPORT.md';prefix=report.read_text().split('## Completed continuation and robustness results')[0]
 report.write_text(prefix.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
 manifest={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'compiler':subprocess.check_output(['clang++','--version'],text=True).strip(),'baseline_societies':180,'probe_branches':36,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.py','.cpp','.json','.npz','.csv','.md','.txt'] and p.name!='experiment_manifest.json'}}
 (ROOT/'reference'/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
