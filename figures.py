from pathlib import Path
import json,base64,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).parent
rows=json.loads((O/'tile-quality.json').read_text());stars={r['tile']:r for r in json.loads((O/'stars.json').read_text())}
valid=[r for r in rows if r['match']['n_external']>0]
f=np.array([r['match']['external_paired_fraction'] for r in valid]);cur=np.array([r['count_source']=='published_manifest' for r in valid])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(11,4.1));bins=np.linspace(.91,1,46)
ax[0].hist(f[~cur],bins=bins,histtype='step',label=f'Recovered counts ({(~cur).sum()})',color='#666666');ax[0].hist(f[cur],bins=bins,histtype='step',label=f'Published metadata ({cur.sum()})',color='#2b6cb0',linewidth=1.8);ax[0].set(yscale='log',xlabel='External paired fraction',ylabel='Distinct published tiles',title='Diagnostic only: no rejection threshold');ax[0].legend(fontsize=8)
for t in [.98,.99]:ax[0].axvline(t,ls=':',lw=1,color='#aa0000');ax[0].text(t,.95,str(t),transform=ax[0].get_xaxis_transform(),ha='right',va='top',fontsize=8)
ax[1].scatter([r['match']['n_external'] for r in valid],1-f,s=13,c=np.where(cur,'#2b6cb0','#aaaaaa'),alpha=.65)
for tile in ['058.220','207.238']:
 r=next(r for r in valid if r['tile']==tile);ax[1].annotate(tile,(r['match']['n_external'],1-r['match']['external_paired_fraction']),xytext=(8,0),textcoords='offset points',fontsize=8)
ax[1].set(xscale='log',xlabel='External catalogue rows',ylabel='External-only fraction',title='The worst nonempty tiles are sparse');fig.tight_layout();fig.savefig(O/'distribution.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(12,7.2),sharex=True)
for ax,key,title in zip(axes.flat[:4],['density_nominal','masked_fraction','flagged_fraction','never_fit_fraction'],['Selected galaxies / nominal arcmin²','Standard-mask fraction','Extraction flags > 2 fraction','Never-fit fraction']):
 for yes,col,lab in [(False,'#aaaaaa','Recovered'),(True,'#2b6cb0','Published')]:
  rs=[r for r in valid if (r['count_source']=='published_manifest')==yes];ax.scatter([r['match']['external_paired_fraction'] for r in rs],[r[key] for r in rs],c=col,s=12,alpha=.65,label=lab)
 ax.set_ylabel(title);ax.axvline(.98,color='#aa0000',ls=':',lw=1)
for i,ax in enumerate(axes.flat[4:],1):
 rs=[r for r in valid if r['n_selected']>=10];ax.errorbar([r['match']['external_paired_fraction'] for r in rs],[r[f'e{i}_mean'] for r in rs],yerr=[r[f'e{i}_se'] for r in rs],fmt='.',ms=2,elinewidth=.5,alpha=.45,color='#555555');ax.axhline(0,c='k',lw=.6);ax.axvline(.98,color='#aa0000',ls=':',lw=1);ax.set_ylabel(f'Mean uncalibrated e{i} (N ≥ 10)');ax.text(.04,.95,'No galaxies below 0.98:\nno shear test there',transform=ax.transAxes,va='top',fontsize=9)
for ax in axes[1]:ax.set_xlabel('External paired fraction')
fig.suptitle('Current mask_v2.0 selection; density ignores masked area and overlaps in its denominator',fontsize=11);fig.tight_layout();fig.savefig(O/'quality-comparison.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,3.6))
for ax,key,title in zip(axes,['de1_rms','de2_rms','dT_T_rms'],['RMS stellar e1 − PSF e1','RMS stellar e2 − PSF e2','RMS (Tstar − Tpsf) / Tstar']):
 rs=[r for r in valid if r['tile'] in stars and stars[r['tile']]['n_star_observations']>1]
 ax.scatter([r['match']['external_paired_fraction'] for r in rs],[stars[r['tile']][key] for r in rs],s=20,c='#2b6cb0');ax.set(xlabel='External paired fraction',ylabel=title)
 for tile in ['058.220','207.238']:
  r=next(r for r in rs if r['tile']==tile);ax.annotate(f"{tile} (N={stars[tile]['n_star_observations']})",(r['match']['external_paired_fraction'],stars[tile][key]),xytext=(3,4),textcoords='offset points',fontsize=7)
fig.suptitle('27 current manifests; per-exposure star observations within each tile sky box',fontsize=11);fig.tight_layout();fig.savefig(O/'psf-residuals.png',dpi=170);plt.close(fig)
cost=[]
for threshold in [.95,.98,.99,.995]:
 rs=[r for r in rows if r['match']['external_paired_fraction']<threshold];cost.append(dict(threshold=threshold,tiles=len(rs),nominal_area_deg2=len(rs)*.25,n_selected=sum(r['n_selected'] for r in rs),selected_fraction=sum(r['n_selected'] for r in rs)/sum(r['n_selected'] for r in rows)))
(O/'threshold-costs.json').write_text(json.dumps(cost,indent=2))
print('COSTS',cost,flush=True)
for lower,upper in [(.98,.99),(.99,.995),(.995,1.00001)]:
 rs=[r for r in valid if lower<=r['match']['external_paired_fraction']<upper];n=sum(r['n_selected'] for r in rs);summary=dict(f_bin=[lower,upper],n_tiles=len(rs),n_gal=n)
 for i in [1,2]:
  mean=sum(r[f'e{i}_sum'] for r in rs)/n if n else 0;var=sum(r[f'e{i}_sumsq'] for r in rs)/n-mean**2 if n else 0;summary[f'e{i}']=[mean,np.sqrt(var/n) if n else None]
 print('ELLIPTICITY',summary,flush=True)
