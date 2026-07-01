"""Reusable confusion-matrix + per-class-F1 plotter (numpy + matplotlib only).
Honest figures for the SHL-2026 paper. Row-normalized (recall) heatmap with counts.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Still","Walk","Run","Bike","Car","Bus","Train","Subway"]  # SHL order, 0-indexed

def per_class_f1(y, p, K=8):
    f1 = np.zeros(K)
    for k in range(K):
        tp = np.sum((p==k)&(y==k)); fp=np.sum((p==k)&(y!=k)); fn=np.sum((p!=k)&(y==k))
        f1[k] = 0.0 if (2*tp+fp+fn)==0 else 2*tp/(2*tp+fp+fn)
    return f1

def confusion(y, p, K=8):
    M = np.zeros((K,K), dtype=int)
    for t,pr in zip(y,p): M[t,pr]+=1
    return M

def plot_cm(y, p, title, out, K=8):
    y=np.asarray(y); p=np.asarray(p)
    M = confusion(y,p,K); Mn = M/np.clip(M.sum(1,keepdims=True),1,None)
    f1 = per_class_f1(y,p,K); macro=f1.mean()
    fig,ax=plt.subplots(figsize=(7.2,6.2))
    im=ax.imshow(Mn,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(CLASSES,rotation=45,ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(K):
        for j in range(K):
            v=Mn[i,j]
            ax.text(j,i,f"{v*100:.0f}",ha="center",va="center",
                    color="white" if v>0.5 else "black",fontsize=8)
    ax.set_title(f"{title}\nmacro-F1={macro:.4f}  (n={len(y)})",fontsize=11)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label("row-normalized (recall)")
    fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)
    return macro, f1

if __name__=="__main__":
    import sys, json
    npz=sys.argv[1]; title=sys.argv[2]; out=sys.argv[3]
    d=np.load(npz, allow_pickle=True)
    y=d["y_true"]; p=d["probabilities"].argmax(1) if "probabilities" in d else d["y_pred"]
    macro,f1=plot_cm(y,p,title,out)
    print(f"macro-F1={macro:.4f}")
    print("per-class F1:", {CLASSES[i]:round(float(f1[i]),3) for i in range(8)})
    print("saved:",out)
