#include <cmath>
#include <algorithm>
#include <cstdint>
extern "C" void evolve(int N,int K,int P,long T,double* w,double* C,double* mu,double* V,const double* X,const double* D,const int64_t* rr,const int64_t* ee,const int64_t* pp,double zfloor,double vfloor,int64_t* counts){
 double cx[256];
 for(long t=0;t<T;t++){
  int r=rr[t],e=ee[t],p=pp[t]; const double* x=X+p*K; double* wr=w+r*K; double* cr=C+r*K*K;
  double er=0,rx=0,xc=0;
  for(int i=0;i<K;i++){er+=w[e*K+i]*x[i];rx+=wr[i]*x[i]; cx[i]=0;for(int j=0;j<K;j++)cx[i]+=cr[i*K+j]*x[j];xc+=x[i]*cx[i];}
  double s=er>=0?1:-1, gc=sqrt(1+xc),v=V[r*N+e],gv=sqrt(1+v), hw=rx*s/gc+D[r*N+e],hm=mu[r*N+e]/gv;
  double pw=0.5*erfc(-hw/sqrt(2.)),pm=0.5*erfc(-hm/sqrt(2.)),z=pw+pm-2*pw*pm;
  if(z<zfloor)counts[1]++; z=std::max(z,zfloor);
  double fw=(1-2*pm)*exp(-hw*hw/2)/sqrt(2*M_PI)/z, fm=(1-2*pw)*exp(-hm*hm/2)/sqrt(2*M_PI)/z;
  double a=-fw*(fw+hw)/(gc*gc),lo=-(1-1e-9)/std::max(xc,1e-300);
  if(a<lo){a=lo; counts[0]++;}
  for(int i=0;i<K;i++){wr[i]+=fw*s/gc*cx[i];for(int j=0;j<K;j++)cr[i*K+j]+=a*cx[i]*cx[j];}
  mu[r*N+e]+=fm/gv*v; double nv=v-fm*(fm+hm)/(gv*gv)*v*v;
  if(nv<vfloor)counts[2]++; V[r*N+e]=std::max(nv,vfloor);
 }
}
