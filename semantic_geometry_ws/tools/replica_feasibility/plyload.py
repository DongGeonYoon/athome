import numpy as np
def load_ply_tris(path):
    raw=open(path,'rb').read(); hdr_end=raw.index(b'end_header\n')+len(b'end_header\n')
    hdr=raw[:hdr_end].decode(); nv=int(hdr.split('element vertex ')[1].split()[0]); nf=int(hdr.split('element face ')[1].split()[0])
    vdt=np.dtype([('p','<f4',3),('n','<f4',3),('c','u1',3)])
    V=np.frombuffer(raw,vdt,nv,hdr_end)['p'].astype(np.float64)
    off=hdr_end+nv*vdt.itemsize; buf=np.frombuffer(raw,np.uint8,offset=off)
    cnt0=buf[0]
    if np.all(buf[::1+4*cnt0][:nf]==cnt0) and len(buf)==nf*(1+4*cnt0):
        F=np.frombuffer(buf.tobytes(),np.dtype([('n','u1'),('i','<i4',int(cnt0))]),nf)['i']
    else: raise RuntimeError('mixed face sizes')
    if F.shape[1]==4: F=np.r_[F[:,[0,1,2]],F[:,[0,2,3]]]
    return V,F.astype(np.int32)
if __name__=='__main__':
    import os
    V,F=load_ply_tris(os.path.expanduser('~/conceptgraph_ws/Replica/room0_mesh.ply')); print(V.shape,F.shape,V.min(0).round(2),V.max(0).round(2))
