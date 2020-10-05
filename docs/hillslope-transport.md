# Contaminant transport on a hillslope

## Import modules


```python
import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.transport import QualityBuilder

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn as sns
```

## Load model data


```python
input_path = '../data/hillslope'
superjunctions = pd.read_csv(f'{input_path}/hillslope_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/hillslope_superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/hillslope_superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/hillslope_superjunction_wq_params.csv')
```

#### Superlink water quality parameters


```python
superlink_wq_params
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K</th>
      <th>D</th>
      <th>c_0</th>
      <th>dx_uk</th>
      <th>dx_dk</th>
      <th>D_uk</th>
      <th>D_dk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0001</td>
      <td>0.001</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1</td>
      <td>0.0001</td>
      <td>0.001</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Superjunction water quality parameters


```python
superjunction_wq_params
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K</th>
      <th>c_0</th>
      <th>bc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.001</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.001</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Instantiate model

The hydraulic model is instantiated with the `SuperLink` class. The water quality model is instantiated with the `QualityBuilder` class.


```python
internal_links = 24
superlink = SuperLink(superlinks, superjunctions,
                      internal_links=internal_links)
waterquality = QualityBuilder(superlink, 
                              superjunction_params=superjunction_wq_params,
                              superlink_params=superlink_wq_params)
```

Specify the hydraulic model parameters, including the default time step `dt`, the inflow into each superjunction `Q_in`, and the inflow into each internal junction `Q_0Ik`.


```python
dt = 10
Q_in = 1e-2 * np.asarray([1., 0.])
Q_0Ik = 1e-3 * np.ones(superlink.NIk)
```

## Run coupled hydraulic / water quality models


```python
c_Ik = []

for _ in range(5000):
    # If time is between 5000 and 10000 s...
    if (superlink.t > 5000) and (superlink.t < 10000):
        # Apply contaminant to uphill superjunction
        c_0j = 10. * np.asarray([1., 0.])
    else:
        # Otherwise, no contaminant input
        c_0j = np.zeros(2)
    # Advance hydraulic model
    superlink.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
    # Advance water quality model
    waterquality.step(dt=dt, c_0j=c_0j)
    # Record nodal contaminant concentrations
    c_Ik.append(waterquality.c_Ik.copy())
```

## Plot time series of contaminant concentrations


```python
sns.set_palette('husl', internal_links + 1)
_ = plt.plot(c_Ik)
plt.ylabel('Concentration $(g/m^3)$')
plt.xlabel('Time (s)')
plt.title('Contaminant concentration time series at each internal junction')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/hillslope-transport/hillslope-transport-0.png)


## Plot profile of ending contaminant concentration


```python
norm = matplotlib.colors.Normalize(vmin=0., vmax=0.625, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='winter')

_ = superlink.plot_profile([0, 1], 
                           superlink_kwargs={'color' : 
                                         [mapper.to_rgba(c) for c in waterquality.c_ik]})
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/hillslope-transport/hillslope-transport-1.png)

