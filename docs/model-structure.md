# Hydraulic model structure

## Primary computational elements

The hydraulic model consists of four primary computational elements:

- **Superjunctions**: Represent reservoir-like structures such as retention basins, ponds, lakes, or manholes. There are *M* superjunctions in a given network, each indexed by *j*. 
- **Superlinks**: Represent pressurized pipes or open channels. Each superlink consists of a chain of *nk+1* junctions and *nk* links connected together in an interleaved fashion. There are *NK* superlinks in a given network, each indexed by *k*.
- **Junctions**: Represent finite volumes within a superlink (e.g. manholes within a sewer line). There are *nk + 1* junctions within each superlink *k*.
- **Links**: Represent segments of pressurized conduit or channel. Each link is bounded by junctions on the upstream and downstream ends. There are *nk* links within each superlink *k*.

The image below shows a stormwater network consisting of 6 superjunctions and 6 superlinks. Each superlink consists of 5 internal junctions and 4 internal links.

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/model-structure/model-structure-0.png)

## Control volumes

Within the hydraulic solver, a staggered-grid discretization scheme is used, in which the continuity equation is applied around each junction and the momentum equation is applied around each link.

### Control volume for continuity equation

<img src="https://pipedream-solver.s3.us-east-2.amazonaws.com/img/model-structure/model-structure-1.png" width=600>


> Note that the left and right edges of the continuity control volume coincide with the midpoints of the upstream and downstream links, respectively.

### Control volume for momentum equation

<img src="https://pipedream-solver.s3.us-east-2.amazonaws.com/img/model-structure/model-structure-2.png" width=600>

> Note that the left and right edges of the momentum control volume coincide with the edges of the upstream and downstream junctions, respectively.

## Indexing schemes

There are four primary indexing schemes used by the hydraulic solver, corresponding to the four primary computational elements.

### Indexed by superjunction (*j*)

State variables that are indexed by superjunction include:

- `H_j` : Superjunction heads (m)
- `A_sj` : Surface area of superjunctions (m^2)
- `V_sj` : Volume stored in superjunctions (m^3)
- `z_inv_j` : Invert elevation of superjunction (m)
- `Q_in` : Exogenous inflow to superjunctions (m^3/s)

For the network shown above, an array corresponding to the superjunction indexing scheme would be of length 6, and appear as:

```
[f_1  f_2  f_3  f_4  f_5  f_6]
```

### Indexed by superlink (*k*)

State variables that are indexed by superlink include:

- `Q_uk` : Flow into upstream end of each superlink (m^3/s)
- `Q_dk` : Flow out of downstream end of each superlink (m^3/s)
- `z_inv_uk` : Invert elevation of upstream end of superlink (m)
- `z_inv_dk` : Invert elevation of downstream end of superlink (m)

For the network shown above, an array corresponding to the superlink indexing scheme would be of length 6, and appear as:

```
[f_1  f_2  f_3  f_4  f_5  f_6]
```

### Indexed by junction (*Ik*)

State variables that are indexed by junction include:

- `h_Ik` : Depth in internal junction *I* within superlink *k* (m)
- `x_Ik` : Horizontal position of internal junction *I* within superlink *k* (m)
- `A_SIk` : Surface area of internal junction *I* within superlink *k* (m^2)

For the network shown above, an array corresponding to the superlink indexing scheme would be of length 30, and appear as:

```
[ f_1,1  f_2,1  f_3,1  f_4,1  f_5,1
  f_1,2  f_2,2  f_3,2  f_4,2  f_5,2
  f_1,3  f_2,3  f_3,3  f_4,3  f_5,3
  f_1,4  f_2,4  f_3,4  f_4,4  f_5,4
  f_1,5  f_2,5  f_3,5  f_4,5  f_5,5
  f_1,6  f_2,6  f_3,6  f_4,6  f_5,6 ]
```

Where the first index corresponds to *I*, the internal junction index, and the second index corresponds to *k*, the superlink index. Note the above array is one-dimensional (read from left to right, then top to bottom).

### Indexed by link (*ik*)

State variables that are indexed by junction include:

- `Q_ik` : Flow in internal link *i* within superlink *k* (m^3/s)
- `A_ik` : Area of flow in internal link *i* within superlink *k* (m^2)
- `dx_ik` : Length of internal link *i* within superlink *k* (m)
- `B_ik` : Top width of flow in internal link *i* within superlink *k* (m)
- `Pe_ik` : Perimeter of flow in internal link *i* within superlink *k* (m)
- `R_ik` : Hydraulic radius of flow in internal link *i* within superlink *k* (m)
- `S_o_ik` : Bottom slope in internal link *i* within superlink *k* (-)

For the network shown above, an array corresponding to the link indexing scheme would be of length 24, and appear as:

```
[ f_1,1  f_2,1  f_3,1  f_4,1
  f_1,2  f_2,2  f_3,2  f_4,2
  f_1,3  f_2,3  f_3,3  f_4,3
  f_1,4  f_2,4  f_3,4  f_4,4
  f_1,5  f_2,5  f_3,5  f_4,5
  f_1,6  f_2,6  f_3,6  f_4,6 ]
```

Where the first index corresponds to *i*, the internal link index, and the second index corresponds to *k*, the superlink index. Note the above array is one-dimensional (read from left to right, then top to bottom).

