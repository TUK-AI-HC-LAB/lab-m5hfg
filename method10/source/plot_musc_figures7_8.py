"""Publication figures 7/8 drawn exclusively from verified local sweep CSVs."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator,FormatStrFormatter

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'results/sweeps_seed42_20261012'
OUT=ROOT/'figures/figures7_8_local';OUT.mkdir(parents=True,exist_ok=True)
def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
proof=json.loads((RUN/'verification.json').read_text());assert proof['status']=='passed'
msm=read(RUN/'msm_interval_ratio_mean_metrics.csv');rscin=read(RUN/'rscin_single_window_mean_metrics.csv')
assert len(msm)==12 and len(rscin)==22
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':12,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.8,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
DATASETS=[('mvtec_ad','MVTec AD',15,1725),('visa','VisA',12,2162)]
colors={'AC':'#7CA6CE','AS':'#EEAB79','single':'#E79A20','without':'#4F78C4','multi':'#CC4C62'}
def decorate(ax):
    ax.grid(axis='y',color='#D8DEE5',linewidth=.6,alpha=.7)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
def export(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300)
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10.6,4.5))
fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.80,wspace=.25)
fig.suptitle('Influence of the minimum-value interval proportion',y=.97,fontsize=14)
fig7=[]
for ax,(ds,title,ncats,nimages) in zip(axes,DATASETS):
    rows=sorted([r for r in msm if r['dataset']==ds],key=lambda r:int(r['percent']))
    assert [int(r['percent']) for r in rows]==[10,30,50,70,90,100]
    assert all(int(r['n_categories'])==ncats and int(r['n_images'])==nimages for r in rows)
    x=np.arange(6);width=.35
    values=[]
    for j,(metric,key) in enumerate([('AC','image_auroc'),('AS','pixel_auroc')]):
        ys=np.array([float(r[key])*100 for r in rows]);values.extend(ys.tolist())
        bars=ax.bar(x+(j-.5)*width,ys,width,color=colors[metric],edgecolor='#58616C',linewidth=.55,label=metric,zorder=3)
        ax.bar_label(bars,labels=[f'{v:.1f}' for v in ys],padding=3,fontsize=7.2,rotation=0,zorder=5)
        for i,(r,y) in enumerate(zip(rows,ys)):
            fig7.append(dict(dataset=ds,metric=metric,percent=int(r['percent']),bar_x=float(x[i]+(j-.5)*width),auroc_percent=float(y)))
    ymin=np.floor(min(values)-1);ymax=np.ceil(max(values)+1)
    ax.set_ylim(ymin,ymax);ax.yaxis.set_major_locator(MultipleLocator(1 if ds=='mvtec_ad' else 2))
    ax.set_xticks(x,[f'{r["percent"]}%' for r in rows]);ax.set_title(title,pad=12)
    ax.set_xlabel('Minimum-value interval proportion');ax.set_ylabel('AUROC (%)')
    decorate(ax)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
fig.text(.5,.105,'AC: Image AUROC   |   AS: Pixel AUROC   |   Bars show category macro means',ha='center',fontsize=8.5)
fig.text(.5,.055,'RTX 5080 · CLIP ViT-L/14-336 · 518-pixel input · seed 42 · local measurements',ha='center',fontsize=8.5,color='#4B5563')
write(OUT/'figure7_data.csv',fig7);export(fig,'figure7_local')

fig,axes=plt.subplots(1,2,figsize=(10.6,4.5))
fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.82,wspace=.25)
fig.suptitle('Influence of RsCIN window mask sizes',y=.97,fontsize=14)
fig8=[]
for ax,(ds,title,ncats,nimages) in zip(axes,DATASETS):
    rows={r['window']:r for r in rscin if r['dataset']==ds}
    assert set(rows)=={'w/o','multi','full',*[str(k) for k in range(2,10)]}
    assert all(int(r['n_categories'])==ncats and int(r['n_images'])==nimages for r in rows.values())
    windows=[str(k) for k in range(2,10)]+['full'];x=np.arange(len(windows))
    ys=np.array([float(rows[k]['image_f1_max'])*100 for k in windows])
    without=float(rows['w/o']['image_f1_max'])*100;multi=float(rows['multi']['image_f1_max'])*100
    ax.plot(x,ys,color=colors['single'],marker='s',markersize=4.5,linewidth=1.6,label='Single window',zorder=3)
    ax.axhline(without,color=colors['without'],linestyle=':',linewidth=1.7,label='w/o RsCIN')
    ax.axhline(multi,color=colors['multi'],linestyle=':',linewidth=1.7,label='Multi-window RsCIN')
    for i,(window,y) in enumerate(zip(windows,ys)):
        fig8.append(dict(dataset=ds,window=window,x=i,single_window_f1_max_percent=float(y),
            without_rscin_f1_max_percent=without,multi_window_f1_max_percent=multi))
    ax.set_xticks(x,[str(k) for k in range(2,10)]+['∞'])
    all_values=ys.tolist()+[without,multi]
    ax.set_ylim(np.floor((min(all_values)-.3)*2)/2,np.ceil((max(all_values)+.3)*2)/2)
    ax.yaxis.set_major_locator(MultipleLocator(.5 if ds=='visa' else 1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(title,pad=12);ax.set_xlabel('Window size k');ax.set_ylabel('Image F1-max (%)')
    decorate(ax)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.06,'∞: all images in the category   |   Category macro means',ha='center',fontsize=8.5)
fig.text(.5,.02,'RTX 5080 · CLIP ViT-L/14-336 · 518-pixel input · seed 42 · local measurements',ha='center',fontsize=8.5,color='#4B5563')
write(OUT/'figure8_data.csv',fig8);export(fig,'figure8_local')

manifest=dict(status='numeric_checks_passed_visual_review_pending',
    dataset_categories={'MVTec AD':15,'VisA':12},dataset_images={'MVTec AD':1725,'VisA':2162},
    figure7_points=24,figure8_single_window_points=18,units='percent; original sweep CSVs are 0–1',
    aggregation='unweighted category macro mean; no pooled dataset metric or error bars',
    figure7_ylim='panel-specific truncated AUROC axes to match the paper comparison style',
    figure8_references={'mvtec_ad_multi_k':[1,2,3],'visa_multi_k':[1,8,9],'infinity':'full category window'},
    data_sources={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'msm_interval_ratio_mean_metrics.csv',RUN/'rscin_single_window_mean_metrics.csv',RUN/'verification.json']},
    plot_source_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.png','.pdf','.svg','.csv']})
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Saved figures and plot data to',OUT)
