# Model inputs

This reference lists model inputs required to run the solvers included in the *pipedream* toolkit:

- [Hydraulic solver inputs](#hydraulic-solver-model-inputs)
- [Infiltration solver inputs](#infiltration-solver-model-inputs)
- [Transport solver inputs](#transport-solver-model-inputs)

---

## Hydraulic solver model inputs

#### Basic model elements
- [Superlinks](#superlinks)
- [Superjunctions](#superjunctions)
- [Links](#links)
- [Junctions](#junctions)

#### Control structures
- [Orifices](#orifices)
- [Weirs](#weirs)
- [Pumps](#pumps)

#### Advanced elements
- [Transects](#transects)
- [Storages](#storages)

---

### Superlinks

<b>`superlinks`</b>: pd.DataFrame
> Table containing all superlinks in the network along with their attributes.

The following fields are required:

|------------|-------|------|-----------------------------------------------------------|
| Field      | Type  | Unit | Description                                               |
|------------|-------|------|-----------------------------------------------------------|
| id         | int   |      | Integer id for the superlink                              |
| name       | str   |      | Name of the superlink                                     |
| sj_0       | int   |      | Index of the upstream superjunction                       |
| sj_1       | int   |      | Index of the downstream superjunction                     |
| in_offset   | float | m    | Offset of superlink invert above upstream superjunction   |
| out_offset | float | m    | Offset of superlink invert above downstream superjunction |
| C_uk       | float | -    | Upstream discharge coefficient                            |
| C_dk       | float | -    | Downstream discharge coefficient                          |
|------------|-------|------|-----------------------------------------------------------|

If internal links and junctions are provided separately, the following fields are required:

|-------|------|------|------------------------------------------|
| Field | Type | Unit | Description                              |
|-------|------|------|------------------------------------------|
| j_0   | int  |      | Index of first junction inside superlink |
| j_1   | int  |      | Index of last junction inside superlink  |
|-------|------|------|------------------------------------------|

If internal links and junctions are not provided separately, the following fields are required:

|-------|-------|-------|------------------------------------------------------|
| Field | Type  | Unit  | Description                                          |
|-------|-------|-------|------------------------------------------------------|
| dx    | float | m     | Length of superlink                                  |
| n     | float | -     | Manning's roughness coefficient for superlink        |
| shape | str   |       | Cross-sectional geometry type (see geometry module)  |
| g1    | float | m     | First dimension of cross-sectional geometry          |
| g2    | float | m     | Second dimension of cross-sectional geometry         |
| g3    | float | m     | Third dimension of cross-sectional geometry          |
| g4    | float | m     | Fourth dimension of cross-sectional geometry         |
| Q_0   | float | m^3/s | Initial flow in internal links                       |
| h_0   | float | m     | Initial depth in internal junctions                  |
| A_s   | float | m     | Surface area of internal junctions                   |
| ctrl  | bool  |       | Indicates presence of control structure in superlink |
| A_c   | float | m^2   | Cross-sectional area of internal control structure   |
| C     | float | -     | Discharge coefficient of internal control structure  |
|-------|-------|-------|------------------------------------------------------|

---

### Superjunctions

<b>`superjunctions`</b>: pd.DataFrame
> Table containing all superjunctions in the network along with their attributes.

The following fields are required:

|-----------|-------|------|-------------------------------------------------------|
| Field     | Type  | Unit | Description                                           |
|-----------|-------|------|-------------------------------------------------------|
| id        | int   |      | Integer id for superjunction                          |
| name      | str   |      | Name of superjunction                                 |
| z_inv     | float | m    | Elevation of bottom of superjunction                  |
| h_0       | float | m    | Initial depth in superjunction                        |
| bc        | bool  |      | Indicates boundary condition at superjunction         |
| storage   | str   |      | Storage type: `functional` or `tabular`               |
| a         | float | m    | `a` value in function relating surface area and depth |
| b         | float | -    | `b` value in function relating surface area and depth |
| c         | float | m^2  | `c` value in function relating surface area and depth |
| max_depth | float | m    | Maximum depth allowed at superjunction                |
|-----------|-------|------|-------------------------------------------------------|

---

### Links

<b>`links`</b>: pd.DataFrame (optional)
> Table containing all links in the network along with their attributes.

Note that if links and junction are not supplied, they will automatically be
generated at even intervals within each superlink.
The following fields are required:

|-------|-------|-------|-----------------------------------------------------|
| Field | Type  | Unit  | Description                                         |
|-------|-------|-------|-----------------------------------------------------|
| j_0   | int   |       | Index of upstream junction                          |
| j_1   | int   |       | Index of downstream junction                        |
| k     | int   |       | Index of containing superlink                       |
| dx    | float | m     | Length of link                                      |
| n     | float | -     | Manning's roughness coefficient for link            |
| shape | str   |       | Cross-sectional geometry type (see geometry module) |
| g1    | float | m     | First dimension of cross-sectional geometry         |
| g2    | float | m     | Second dimension of cross-sectional geometry        |
| g3    | float | m     | Third dimension of cross-sectional geometry         |
| g4    | float | m     | Fourth dimension of cross-sectional geometry        |
| Q_0   | float | m^3/s | Initial flow in internal links                      |
| h_0   | float | m     | Initial depth in internal junctions                 |
| A_s   | float | m     | Surface area of internal junctions                  |
| ctrl  | bool  |       | Indicates presence of control structure in link     |
| A_c   | float | m^2   | Cross-sectional area of internal control structure  |
| C     | float | -     | Discharge coefficient of internal control structure |
|-------|-------|-------|-----------------------------------------------------|

---

### Junctions

<b>`junctions`</b>: pd.DataFrame (optional)
> Table containing all junctions in the network along with their attributes.

Note that if links and junction are not supplied, they will automatically be
generated at even intervals within each superlink.
The following fields are required:

|-------|-------|------|-------------------------------|
| Field | Type  | Unit | Description                   |
|-------|-------|------|-------------------------------|
| id    | int   |      | Integer id for junction       |
| k     | int   |      | Index of containing superlink |
| h_0   | float | m    | Initial depth at junction     |
| A_s   | float | m^2  | Surface area of junction      |
| z_inv | float | m    | Invert elevation of junction  |
|-------|-------|------|-------------------------------|

---

### Orifices

<b>`orifices`</b>: pd.DataFrame (optional)
> Table containing orifice control structures, and their attributes.

The following fields are required:

|-------------|-------|------|------------------------------------------------------|
| Field       | Type  | Unit | Description                                          |
|-------------|-------|------|------------------------------------------------------|
| id          | int   |      | Integer id for the orifice                           |
| name        | str   |      | Name of the orifice                                  |
| sj_0        | int   |      | Index of the upstream superjunction                  |
| sj_1        | int   |      | Index of the downstream superjunction                |
| orientation | str   |      | Orifice orientation: `bottom` or `side`              |
| C           | float | -    | Discharge coefficient for orifice                    |
| A           | float | m^2  | Full area of orifice                                 |
| y_max       | float | m    | Full height of orifice                               |
| z_o         | float | m    | Offset of bottom above upstream superjunction invert |
|-------------|-------|------|------------------------------------------------------|

---

### Weirs

<b>`weirs`</b>: pd.DataFrame (optional)
> Table containing weir control structures, and their attributes.

The following fields are required:

|-------|-------|------|------------------------------------------------------|
| Field | Type  | Unit | Description                                          |
|-------|-------|------|------------------------------------------------------|
| id    | int   |      | Integer id for the weir                              |
| name  | str   |      | Name of the weir                                     |
| sj_0  | int   |      | Index of the upstream superjunction                  |
| sj_1  | int   |      | Index of the downstream superjunction                |
| z_w   | float | m    | Offset of bottom above upstream superjunction invert |
| y_max | float | m    | Full height of weir                                  |
| C_r   | float | -    | Discharge coefficient for rectangular portion        |
| C_t   | float | -    | Discharge coefficient for triangular portions        |
| L     | float | m    | Length of rectangular portion of weir                |
| s     | float | -    | Inverse slope of triangular portion of weir          |
|-------|-------|------|------------------------------------------------------|

---

### Pumps

<b>`pumps`</b>: pd.DataFrame (optional)
> Table containing pump control structures and their attributes.

The following fields are required:

|--------|-------|------|------------------------------------------------------|
| Field  | Type  | Unit | Description                                          |
|--------|-------|------|------------------------------------------------------|
| id     | int   |      | Integer id for the pump                              |
| name   | str   |      | Name of the pump                                     |
| sj_0   | int   |      | Index of the upstream superjunction                  |
| sj_1   | int   |      | Index of the downstream superjunction                |
| z_p    | float | m    | Offset of bottom above upstream superjunction invert |
| a_q    | float |      | Vertical coefficient of pump ellipse                 |
| a_h    | float |      | Horizontal coefficient of pump ellipse               |
| dH_min | float | m    | Minimum pump head                                    |
| dH_max | float | m    | Maximum pump head                                    |
|--------|-------|------|------------------------------------------------------|

---

### Transects

<b>`transects`</b>: dict (optional)
> Dictionary describing nonfunctional channel cross-sectional geometries.

Takes the following structure:

    {
        <transect_name> :
            {
                'x' : <x-coordinates of cross-section (list of floats)>,
                'y' : <y-coordinates of cross-section (list of floats)>,
                'horiz_points' : <Number of horizontal sampling points (int)>
                'vert_points' : <Number of vertical sampling points (int)>
            }
        ...
    }

---

### Storages

<b>`storages`</b>: dict (optional)
> Dictionary describing tabular storages for superjunctions.

Takes the following structure:

    {
        <storage_name> :
            {
                'h' : <Depths (list of floats)>,
                'A' : <Surface areas associated with depths (list of floats)>,
            }
        ...
    }

## Infiltration solver model inputs

- [Soil Parameters](#soil-parameters)

---

### Soil parameters

<b>`soil_params`</b>: pd.DataFrame
> Table containing soil parameters for all catchments.

The following fields are required.

|---------|-------|------|------------------------------------------------------|
| Field   | Type  | Unit | Description                                          |
|---------|-------|------|------------------------------------------------------|
| psi_f   | float | m    | Matric potential of the wetting front (suction head) |
| Ks      | float | m/s  | Saturated hydraulic conductivity                     |
| theta_s | float | -    | Saturated soil moisture content                      |
| theta_i | float | -    | Initial soil moisture content                        |
| A_s     | float | m^2  | Surface area of soil element                         |
|---------|-------|------|------------------------------------------------------|

---

## Transport solver model inputs

- [Superlink parameters](#superlink-parameters)
- [Superjunction parameters](#superjunction-parameters)
- [Link parameters](#link-parameters)
- [Junction parameters](#junction-parameters)

---

### Superlink parameters

<b>`superlink_params`</b>: pd.DataFrame
> Table containing superlink water quality parameters.

The following fields are required:

|---------|-------|-------|------------------------------------------------------------------------|
| Field   | Type  | Unit  | Description                                                            |
|---------|-------|-------|------------------------------------------------------------------------|
| dx_uk   | float | m     | Distance from the upstream end of the superlink to well-mixed region   |
| dx_dk   | float | m     | Distance from the downstream end of the superlink to well-mixed region |
| D_uk    | float | m     | Diffusion constant of upstream end of superlink                        |
| D_dk    | float | m     | Diffusion constant of downstream end of superlink                      |
|---------|-------|-------|------------------------------------------------------------------------|

If internal links and junctions are not provided separately, the following fields are required:

|---------|-------|-------|------------------------------------------------------------------------|
| Field   | Type  | Unit  | Description                                                            |
|---------|-------|-------|------------------------------------------------------------------------|
| K       | float | 1/s   | First order reaction constant in internal links/junctions              |
| D       | float | m^2/s | Diffusion constant in internal links/junctions                         |
| c_0     | float | g/m^3 | Initial contaminant concentration in internal links/junctions          |
|---------|-------|-------|------------------------------------------------------------------------|

---

### Superjunction parameters

<b>`superjunction_params`</b>: pd.DataFrame
> Table containing superjunction water quality parameters.

The following fields are required:

|-------|-------|-------|-----------------------------------------------------------|
| Field | Type  | Unit  | Description                                               |
|-------|-------|-------|-----------------------------------------------------------|
| K     | float | 1/s   | First order reaction constant in superjunction            |
| c_0   | float | g/m^3 | Initial contaminant concentration in superjunction        |
| bc    | bool  | -     | Indicates contaminant boundary condition at superjunction |
|-------|-------|-------|-----------------------------------------------------------|

---

### Link parameters

<b>`link_params`</b>: pd.DataFrame (optional)
> Table containing link water quality parameters.

The following fields are required:

|-------|-------|-------|-------------------------------------------|
| Field | Type  | Unit  | Description                               |
|-------|-------|-------|-------------------------------------------|
| K     | float | 1/s   | First order reaction constant in link     |
| D     | float | m^2/s | Diffusion constant in link                |
| c_0   | float | g/m^3 | Initial contaminant concentration in link |
|-------|-------|-------|-------------------------------------------|

---

### Junction parameters

<b>`junction_params`</b>: pd.DataFrame (optional)
> Table containing junction water quality parameters.

The following fields are required:

|-------|-------|-------|-----------------------------------------------|
| Field | Type  | Unit  | Description                                   |
|-------|-------|-------|-----------------------------------------------|
| K     | float | 1/s   | First order reaction constant in junction     |
| D     | float | m^2/s | Diffusion constant in junction                |
| c_0   | float | g/m^3 | Initial contaminant concentration in junction |
|-------|-------|-------|-----------------------------------------------|

