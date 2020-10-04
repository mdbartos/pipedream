# Governing equations

This page describes the mathematical basis for the *pipedream* toolkit. The governing equations for the hydraulic, hydrologic, and water quality solvers are described, along with their discretization schemes.

## Hydraulic solver

The hydraulic solver is based on the one-dimensional Saint Venant equations for unsteady flow. This coupled pair of hyperbolic partial differential equations consists of a mass balance and a momentum balance.

\begin{equation} \frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_{in} \end{equation}

\begin{equation} \frac{\partial Q}{\partial t} + \frac{\partial}{\partial x} (Q u) + g A \biggl(\frac{\partial h}{\partial x} - S_0 + S_f + S_L \biggr) = 0 \end{equation}

Where \\(Q\\) is discharge; \\(A\\) is the cross-sectional area of flow; \\(u\\) is the
average velocity; \\(h\\) is depth; \\(x\\) is distance; \\(t\\) is time; \\(q_{in}\\) is the
lateral inflow per unit width; and \\(S_o\\), \\(S_f\\) and \\(S_L\\) represent the channel
bottom slope, friction slope and local head loss slope, respectively.

The method used by *pipedream* to solve these equations is based on the
SUPERLINK scheme posed by Ji[^1]. Using a staggered grid formulation, the
continuity equation is applied to each junction (indexed by \\(Ik\\)), while the
momentum equation applied to each link (indexed by \\(ik\\)). The equations are
discretized using a Backward Euler-type implicit scheme:

\begin{equation} Q_{ik}^{t + \Delta t} - Q_{i - 1k}^{t + \Delta t} + \biggl( \frac{B_{ik} \Delta x_{ik}}{2} + \frac{B_{i - 1k} \Delta x_{i - 1k}}{2} + A_{s,Ik} \biggr) \cdot \frac{h_{Ik}^{t + \Delta t} - h_{Ik}^t}{\Delta t} = Q_{in,Ik} \end{equation}

\begin{equation}
  \begin{split}
    (Q_{ik}^{t + \Delta t} - Q_{ik}^t) \frac{\Delta x_{ik}}{\Delta t} + u_{I+1k}
    Q_{I + 1k}^{t + \Delta t} - u_{Ik} Q_{Ik}^{t + \Delta t} \\
    + g A_{ik} (h_{I+1k}^{t + \Delta t} - h_{Ik}^{t + \Delta t}) 
    - g A_{ik} S_{o,ik} \Delta x_{ik} + g
    A_{ik} (S_{f,ik} + S_{L,ik}) \Delta x = 0
  \end{split}
\end{equation}

Where \\(B\\) is the top width of flow, \\(A_s\\) is the junction surface area, and
\\(Q_{in}\\) is the exogenous flow input.

The boundary conditions for each superlink are supplied by the upstream and
downstream superjunction heads, assuming weir-like flow at the superlink inlet
and outlet:

\begin{equation}
  Q = C A \sqrt{2 g \Delta H} 
\end{equation}

Where \\(C\\) is the inlet/outlet discharge coefficient, and \\(\Delta H\\) is the
difference in head between the superjunction and the adjacent superlink boundary
junction.

The hydraulic model solves for all unknowns simultaneously at each time step by
embedding the solutions to the Saint-Venant equations into a system of implicit
linear equations wherein all unknowns are defined in terms of the unknown
superjunction heads.

- First, the discretized Saint-Venant equations are reformulated into recurrence
  relations that relate junction heads and link flows within each superlink.
- The assumption of orifice-like flow between superjunctions and superlinks
  is used to establish boundary conditions for the superlink inlets and outlets.
- Combining the recurrence relations together with the superlink boundary
  conditions, the system is reformulated as a sparse matrix equation with all
  unknowns expressed in terms of the unknown superjunction heads.
- After solving for the unknown superjunction heads, the internal depths and
  flows within each superlink are recovered by substituting the superjunction
  boundary heads into the previously-developed recurrence relations.
  
## Infiltration solver

Infiltration and runoff are computed using the Green-Ampt method. At each time
step, the integrated form of the Green-Ampt equation is solved to estimate the
cumulative infiltration depth for a soil element (indexed by \\(f\\)).

\begin{equation}
  h_f^{t + \Delta t}  = K_s \Delta t + h_f^t + \psi_f \theta_d \biggl( \log(h_f^{t + \Delta t} + \psi_f \theta_d) - \log(h_f^t + \psi_f \theta_d) \biggr)
\end{equation}

Where \\(h_f^{t + \Delta t}\\) is the cumulative infiltration depth at time \\(t +
\Delta t\\) (m), \\(K_s\\) is the saturated hydraulic conductivity (m/s), \\(\psi_f\\) is
the suction head of the wetting front (m), and \\(\theta_d\\) is the soil moisture
deficit (unitless). The infiltration rate, \\(i_f\\) (m/s) is then estimated as:

\begin{equation}
 i_f^{t + \Delta t} = \frac{h_f^{t + \Delta t} - h_f^t}{\Delta t}
\end{equation}

The runoff rate per unit area is equal to the precipitation rate minus the infiltration rate:

\begin{equation}
 q_f^{t + \Delta t} = p_f^{t + \Delta t} - i_f^{t + \Delta t}
\end{equation}

Where \\(q_f^{t + \Delta t}\\) is the runoff rate per unit area (m/s), and \\(p_f\\) is
the precipitation rate per unit area (m/s).

The process of soil recovery (by which soil gradually dries due to evaporation
and drainage), is implemented using the empirical method of Huber et al. (2005)
[^2].

## Water quality solver

The transport of a contaminant through a control volume like a stormwater pipe
is described by the one-dimensional advection-reaction-diffusion equation:

\begin{equation}
  \frac{\partial c}{\partial t} + \frac{\partial (u c)}{\partial x} - \frac{\partial }{\partial x} \biggl( D \frac{\partial c}{\partial x} \biggr) - r(c) = 0
\end{equation}

Where \\(c\\) is the concentration of the contaminant, \\(u\\) is the velocity of flow,
\\(D\\) is the diffusion coefficient, \\(r(c)\\) is the endogenous reaction rate, \\(t\\) is
time and \\(x\\) is distance.

To solve this equation, one must supply a set of boundary conditions. For a
stormwater system consisting of junctions connected by conduits, the boundary
conditions for a conduit are given by the concentrations at the upstream and
downstream junctions. Applying the continuity equation to the mass of the
contaminant, the concentration at these junctions is given by the sum of inflows
minus outflows minus any loss of material due to reactions:

\begin{equation}
    \frac{d (Vc)}{dt} = Q_{u} c_{u} - Q_{d} c_{d} + Q_{o} c_{o} - V r(c)
\end{equation}

Where \\(c\\) is the concentration of the contaminant in the junction, \\(V\\) is the
volume of solution in the junction, \\(Q_u\\) and \\(Q_d\\) are the upstream and
downstream volumetric flow rates, \\(c_u\\) and \\(c_d\\) are the upstream and
downstream concentrations, \\(Q_o\\) is the exogenous lateral inflow, \\(c_o\\) is the
concentration of the exogenous lateral inflow, and \\(r(c)\\) is the endogenous
reaction rate.

The advection-reaction-diffusion equation is discretized using a Backward Euler
scheme, and applied to each link \\(ik\\):

\begin{equation}
    \frac{1}{\Delta t} (c_{ik}^{t + \Delta t} - c_{ik}^t) + \frac{1}{\Delta x_{ik}} \biggl( u_{I + 1k} c_{I + 1k}^{t + \Delta t} - u_{Ik} c_{Ik}^{t + \Delta t} \biggr) - D_{ik} \biggl( \frac{c_{I + 1k}^{t + \Delta t} - c_{ik}^{t + \Delta t}}{\frac{1}{2} \Delta x_{ik}^2} - \frac{c_{ik}^{t + \Delta t} - c_{Ik}^{t + \Delta t}}{\frac{1}{2} \Delta x_{ik}^2}\biggr) - K_{ik} c_{ik}^{t + \Delta t} = 0
\end{equation}

<!-- \begin{equation} -->
<!--     \frac{1}{\Delta t} (\bar{c}_{ik}^{t + \Delta t} - \bar{c}_{ik}^t) + \frac{1}{\Delta x_{ik}} \biggl( u_{I + 1k} c_{I + 1k}^{t + \Delta t} - u_{Ik} c_{Ik}^{t + \Delta t} \biggr) - D_{ik} \biggl( \frac{c_{I + 1k}^{t + \Delta t} - \bar{c}_{ik}^{t + \Delta t}}{\frac{1}{2} \Delta x_{ik}^2} - \frac{\bar{c}_{ik}^{t + \Delta t} - c_{Ik}^{t + \Delta t}}{\frac{1}{2} \Delta x_{ik}^2}\biggr) - K_{ik} \bar{c}_{ik}^{t + \Delta t} = 0 -->
<!-- \end{equation} -->

Where \\(c_{ik}\\) is the concentration in the link, \\(c_{Ik}\\) and \\(c_{I+1k}\\)
are the concentrations at the upstream and downstream junctions, \\(u_{Ik}\\) and
\\(u_{I+1k}\\) are the flow velocities at the upstream and downstream junctions,
\\(D_{ik}\\) is the diffusion coefficient in the conduit, \\(\Delta x_{ik}\\) is the
length of the conduit, and \\(\Delta t\\) is the time step. Here, it is assumed that
the reaction rate can be represented by a first-order reaction with constant
\\(K_1\\):

\begin{equation}
    r(c) = K_{ik} c
\end{equation}

Using a staggered-grid formulation, the mass conservation equation is applied
around each junction \\(Ik\\):

\begin{equation}
  c_{ik}^{t + \Delta t} Q_{ik}^{t + \Delta t} - c_{i-1k}^{t + \Delta t} Q_{i-1k}^{t + \Delta t} + \frac{A_{s, Ik}}{\Delta t} (c_{Ik}^{t + \Delta t} h_{Ik}^{t + \Delta t} - c_{Ik}^t h_{Ik}^t) \\\\ + \frac{B_{ik} \Delta x_{ik}}{2 \Delta t} (c_{ik}^{t + \Delta t} h_{Ik}^{t + \Delta t} - c_{ik}^t h_{Ik}^t) + \frac{B_{i-1k} \Delta x_{i-1k}}{2 \Delta t} (c_{i-1k}^{t + \Delta t} h_{Ik}^{t + \Delta t} - c_{i-1k}^t h_{Ik}^t) \\\\ = c_{0,Ik}^{t + \Delta t} Q_{0, Ik}^{t + \Delta t} - (K_{Ik} A_{s,Ik} c_{Ik}^{t + \Delta t} h_{Ik}^{t + \Delta t} + \frac{\Delta x_{ik}}{2} K_{ik} A_{ik} c_{ik}^{t + \Delta t} + \frac{\Delta x_{i-1k}}{2} K_{i-1k} A_{i-1k} c_{i-1k}^{t + \Delta t})
\end{equation}

<!-- \begin{equation} -->
<!--   \bar{c}_{ik}^{t + \Delta t} Q_{ik}^{t + \Delta t} - \bar{c}_{i-1k}^{t + \Delta t} Q_{i-1k}^{t + \Delta t} + \frac{A_{s, Ik}}{\Delta t} (c_{Ik}^{t + \Delta t} h_{Ik}^{t + \Delta t} - c_{Ik}^t h_{Ik}^t) + \frac{B_{ik} \Delta x_{ik}}{2 \Delta t} (\bar{c}_{ik}^{t + \Delta t} h_{Ik}^{t + \Delta t} - \bar{c}_{ik}^t h_{Ik}^t) + \frac{B_{i-1k} \Delta x_{i-1k}}{2 \Delta t} (\bar{c}_{i-1k}^{t + \Delta t} h_{Ik}^{t + \Delta t} - \bar{c}_{i-1k}^t h_{Ik}^t) = c_{0,Ik}^{t + \Delta t} Q_{0, Ik}^{t + \Delta t} - (K_{Ik} A_{s,Ik} c_{Ik}^{t + \Delta t} h_{Ik}^{t + \Delta t} + \frac{\Delta x_{ik}}{2} K_{ik} A_{ik} c_{ik}^{t + \Delta t} + \frac{\Delta x_{i-1k}}{2} K_{i-1k} A_{i-1k} c_{i-1k}^{t + \Delta t}) -->
<!-- \end{equation} -->

Where \\(Q_{ik}\\) is the flow rate in link \\(i\\), \\(h_{Ik}\\) is the depth in junction
\\(Ik\\), \\(A_{s,Ik}\\) is the surface area of junction \\(Ik\\), \\(B_{ik}\\) is the top width
of flow in link \\(ik\\), \\(Q_{0,Ik}\\) is the lateral overflow into the control volume,
and \\(c_{0,Ik}\\) is the contaminant concentration in the lateral overflow.

## References

[^1]: Ji, Z. (1998). General Hydrodynamic Model for Sewer/Channel Network Systems. Journal of Hydraulic Engineering, 124(3), 307–315. doi: 10.1061/(asce)0733-9429(1998)124:3(307)

[^2]: W. Huber, R. Dickinson, L. Rossman, EPA storm water management model, SWMM5, in: Watershed Models, CRC Press, 2005, pp. 338–359. doi:10.1201/9781420037432.ch14
