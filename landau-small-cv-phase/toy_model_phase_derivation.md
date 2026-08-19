# Toy-Model Phase Boundaries in the \(C,V\ll 1\) Limit

The full model uses the scaled fields

\[
h_w=\frac{\sigma_e\,\hat w\cdot x}{\sqrt{1+x^\top Cx}},
\qquad
h_\mu=\frac{\mu_{e|r}}{\sqrt{1+V}}.
\]

For small but nonzero \(C\) and \(V\),

\[
h_w \simeq \sigma_e\,\hat w\cdot x,
\qquad
h_\mu \simeq \mu_{e|r},
\]

while \(C\) and \(V\) mainly multiply the ideological and affective update rates. Thus, to leading order, they set time scales rather than the fixed-point phase boundaries.

We then introduce two coarse-grained order parameters:

- \(M\): class-affective order, analogous to \(R_{\mu,c}\)
- \(Q\): class-ideological order, analogous to \(R_{c,w}\)

and an effective collective discrimination strength

\[
g=\kappa f_d d,
\]

where \(f_d\) is the fraction of discriminatory agents, \(d\) is the discrimination field, and \(\kappa\) absorbs details of the discrimination matrix and group composition.

A minimal phenomenological model is

\[
\dot M
=
V\left[a(g-g_c)M-bM^3\right],
\]

\[
\dot Q
=
C\left[(\lambda M-\chi M^2)Q-uQ^3\right].
\]

For \(M\), the nonzero fixed point appears when \(g>g_c\):

\[
M^2=\frac{a}{b}(g-g_c).
\]

Therefore the neutral-to-discriminatory boundary is

\[
d_1(f_d)=\frac{g_c}{\kappa f_d}.
\]

Ideological order exists while

\[
0<M<\frac{\lambda}{\chi}.
\]

The transition to the strongly class-driven regime occurs at

\[
M_*=\frac{\lambda}{\chi}.
\]

Substituting this into the fixed-point equation for \(M\) gives

\[
g_2
=
g_c+\frac{b}{a}\left(\frac{\lambda}{\chi}\right)^2,
\]

and therefore

\[
d_2(f_d)
=
\frac{
g_c+\frac{b}{a}\left(\frac{\lambda}{\chi}\right)^2
}{\kappa f_d}.
\]

For negative \(d\), an analogous phenomenological threshold gives

\[
d_-(f_d)
=
-\frac{g_c^-}{\kappa_- f_d}.
\]

The resulting schematic phases are

\[
\begin{array}{rcl}
d<d_- &:& \text{I: tolerant / spin-glass},\\
d_-<d<d_1 &:& \text{II: neutral},\\
d_1<d<d_2 &:& \text{III: discriminatory + ideological},\\
d>d_2 &:& \text{IV: strongly class-driven}.
\end{array}
\]

For the illustrative plot, the parameters were chosen only to make all four regions visible:

\[
g_c=g_c^-=0.20,\quad
a=b=1,\quad
\lambda=0.5,\quad
\chi=1,\quad
\kappa=\kappa_-=1.
\]

This gives

\[
d_-=-\frac{0.20}{f_d},
\qquad
d_1=\frac{0.20}{f_d},
\qquad
d_2=\frac{0.45}{f_d}.
\]

These numerical values were **not fitted to the simulations**. They are illustrative. A quantitative version should estimate the three effective thresholds from the microscopic model or simulation data and test whether the boundaries are indeed approximately controlled by \(f_d d=\text{constant}\).
