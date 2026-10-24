import argparse,json,os,time,platform,shutil
from pathlib import Path
from . import data,ehr,movement,simulation,figures,report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='data/raw');parser.add_argument('--output',default='results');parser.add_argument('--stage',choices=['baseline','sepsis','all']);args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);root=Path(args.data_root).resolve()
    cfg=json.loads(Path('configs/protocol.json').read_text());start=time.time()
    provenance={'run_id':os.environ.get('ORX_RUN_ID','manual'),'commit':os.environ.get('ORX_COMMIT','manual'),'python':platform.python_version(),'configuration':cfg,'scope':'EHR + consent/oversight primary; skeletons secondary; no clinical validation'}
    stage=args.stage or (json.loads(Path('configs/execution.json').read_text())['stage'] if Path('configs/execution.json').exists() else 'baseline')
    if stage in ['sepsis','all']:
        from . import sepsis,sepsis_reporting
        dest=out/'sepsis';dest.mkdir(exist_ok=True);scfg=json.loads(Path('configs/sepsis_protocol.json').read_text())
        sp={**provenance,'configuration':scfg,'stage':'sepsis'};(dest/'provenance.json').write_text(json.dumps(sp,indent=2));print('EFFECTIVE_CONFIGURATION',json.dumps(sp),flush=True)
        sepsis.run(root,dest,scfg);gd=dest/'graph';gd.mkdir(exist_ok=True);graphs,_=simulation.structural(gd)
        figures.diagrams(graphs,gd/'figures');sepsis_reporting.create(dest,scfg,sp)
        print('SEPSIS_FINAL_SUMMARY',json.dumps({'status':'complete','output':str(dest),'seconds':time.time()-start}),flush=True)
        if stage=='sepsis':return
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2));print('EFFECTIVE_CONFIGURATION',json.dumps(provenance),flush=True)
    data.acquire(root,out)
    full=root/'mimiciv';ehrroot=full if (full/'hosp/patients.csv.gz').exists() or (full/'hosp/patients.csv').exists() else root/'mimic-demo'
    cohort=ehr.build(ehrroot,cfg['ehr'],out);pred,em=ehr.evaluate(cohort,cfg,out);ehr.provider_graph(ehrroot,cohort,out)
    graphs,gm=simulation.structural(out);cap,queues=simulation.workload(pred,cfg,out);simulation.monte_carlo(cfg,out)
    mp,mm=movement.evaluate(root/'upfall-skeleton',cfg,out);simulation.integrated_pairing(mp,pred,cfg,out)
    figdir=out/'figures';figures.diagrams(graphs,figdir);figures.risk_capacity(cap,queues,figdir)
    # Keep the complete plotting script and style beside the figures for correction.
    shutil.copy('consent_ai/figures.py',figdir/'figures_source.py');shutil.copy('consent_ai/orx_figstyle.py',figdir/'orx_figstyle.py')
    report.write_report(out,cfg,provenance);figures.dashboard(out,out/'index.html')
    print('FINAL_SUMMARY',json.dumps({'seconds':time.time()-start,'ehr_metric_rows':len(em),'movement_metric_rows':len(mm),'output':str(out),'status':'completed public-data and simulation scope','clinical_validation':False}),flush=True)

if __name__=='__main__':main()
