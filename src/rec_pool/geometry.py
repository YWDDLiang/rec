"""Dictionary learning as in the supplied PDF; added geometry checks are explicit."""
from __future__ import annotations
import numpy as np


def matrix(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('expected finite nonempty matrix')
    return x


def unit(x):
    x=matrix(x);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30)


def omp(x,d,sparsity=4):
    x=matrix(x);d=matrix(d)
    if x.shape[1]!=d.shape[1] or type(sparsity) is not int or not 1<=sparsity<=len(d):
        raise ValueError('invalid dictionary/sparsity')
    norm=np.linalg.norm(d,axis=1)
    if np.any(norm<1e-12):raise ValueError('zero atom')
    c=np.zeros((len(x),len(d)))
    for i,row in enumerate(x):
        residual=row.copy();active=[]
        for _ in range(sparsity):
            corr=np.abs(d@residual)/norm;corr[active]=-np.inf
            j=int(np.argmax(corr))
            if corr[j]<1e-12:break
            active.append(j);w=np.linalg.lstsq(d[active].T,row,rcond=1e-10)[0]
            residual=row-w@d[active]
            if np.linalg.norm(residual)<1e-10:break
        if active:c[i,active]=w
    return c


def fit_dictionary(raw,n_atoms=64,sparsity=4,fit_limit=8192,seed=42,max_iter=1000):
    from sklearn.decomposition import MiniBatchDictionaryLearning
    x=unit(raw)
    if type(n_atoms) is not int or not 1<=n_atoms<=min(len(x),x.shape[1],fit_limit):
        raise ValueError('too many atoms for fit set/dimension')
    if not 1<=sparsity<=n_atoms:raise ValueError('invalid sparsity')
    rng=np.random.default_rng(seed)
    ids=np.sort(rng.choice(len(x),min(fit_limit,len(x)),replace=False))
    model=MiniBatchDictionaryLearning(n_components=n_atoms,alpha=1.,batch_size=min(256,len(ids)),
        max_iter=max_iter,transform_algorithm='omp',transform_n_nonzero_coefs=sparsity,
        random_state=seed,n_jobs=1)
    model.fit(x[ids]);d=unit(model.components_)
    return d,omp(x,d,sparsity),ids


def coordinates(c,d):
    """Return Z s.t. Z Z^T = C D D^T C^T, NOT generally C C^T."""
    c=matrix(c);d=matrix(d)
    if c.shape[1]!=len(d):raise ValueError('code/dictionary mismatch')
    eig,v=np.linalg.eigh(d@d.T)
    return c@(v*np.sqrt(np.maximum(eig,0.))[None,:])


def alignment(c,d,ref,cosine=True):
    c=matrix(c);d=matrix(d);ref=matrix(ref)
    if c.shape[1]!=len(d) or ref.shape[1]!=d.shape[1]:raise ValueError('reference mismatch')
    z=c@d
    return (unit(z)@unit(ref).T) if cosine else z@ref.T


def diagnostics(raw,c,d):
    x=unit(raw);c=matrix(c);d=matrix(d);recon=c@d
    share=np.max(np.abs(c),axis=1)/np.maximum(np.sum(np.abs(c),axis=1),1e-30)
    vals=np.maximum(np.linalg.eigvalsh(x.T@x/len(x)),0)
    p=vals/max(vals.sum(),1e-30);p=p[p>0]
    return {'rows':len(x),'atoms':len(d),'dominant_share_mean':float(share.mean()),
        'dominant_share_ge_0_7':float(np.mean(share>=.7)),
        'reconstruction_error_mean':float(np.linalg.norm(x-recon,axis=1).mean()),
        'active_count_mean':float((np.abs(c)>1e-10).sum(1).mean()),
        'effective_rank_proxy':float(np.exp(-np.sum(p*np.log(p)))),
        'gram_offdiagonal_norm':float(np.linalg.norm(d@d.T-np.diag(np.diag(d@d.T)))),
        'scope':'probe geometry only; not number of true interests or recommendation quality'}
