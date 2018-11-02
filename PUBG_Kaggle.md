

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
```


```python
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



```python
from sklearn import model_selection
from sklearn.metrics import accuracy_score
```


```python
#import training data
PG_train = pd.read_csv("train_V2.csv")
```


```python
print(f'There are {PG_train.shape[0]} samples and {PG_train.shape[1]} features in the training PUBG V2 dataset.')
```

    There are 4446966 samples and 29 features in the training PUBG V2 dataset.



```python
pd.set_option('display.max_columns', None)
PG_train.head()
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
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>killStreaks</th>
      <th>longestKill</th>
      <th>matchDuration</th>
      <th>matchType</th>
      <th>maxPlace</th>
      <th>numGroups</th>
      <th>rankPoints</th>
      <th>revives</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7f96b2f878858a</td>
      <td>4d4b580de459be</td>
      <td>a10357fd1a4a91</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>1241</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>1306</td>
      <td>squad-fpp</td>
      <td>28</td>
      <td>26</td>
      <td>-1</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>244.80</td>
      <td>1</td>
      <td>1466</td>
      <td>0.4444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>eef90569b9d03c</td>
      <td>684d5656442f9e</td>
      <td>aeb375fc57110c</td>
      <td>0</td>
      <td>0</td>
      <td>91.47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>1777</td>
      <td>squad-fpp</td>
      <td>26</td>
      <td>25</td>
      <td>1484</td>
      <td>0</td>
      <td>0.0045</td>
      <td>0</td>
      <td>11.04</td>
      <td>0</td>
      <td>0</td>
      <td>1434.00</td>
      <td>5</td>
      <td>0</td>
      <td>0.6400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1eaf90ac73de72</td>
      <td>6a4a42c3245a74</td>
      <td>110163d8bb94ae</td>
      <td>1</td>
      <td>0</td>
      <td>68.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>1318</td>
      <td>duo</td>
      <td>50</td>
      <td>47</td>
      <td>1491</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>161.80</td>
      <td>2</td>
      <td>0</td>
      <td>0.7755</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4616d365dd2853</td>
      <td>a930a9c79cd721</td>
      <td>f1f1f4ef412d7e</td>
      <td>0</td>
      <td>0</td>
      <td>32.90</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>1436</td>
      <td>squad-fpp</td>
      <td>31</td>
      <td>30</td>
      <td>1408</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>202.70</td>
      <td>3</td>
      <td>0</td>
      <td>0.1667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>315c96c26c9aac</td>
      <td>de04010b3458dd</td>
      <td>6dc8ff871e21e6</td>
      <td>0</td>
      <td>0</td>
      <td>100.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>58.53</td>
      <td>1424</td>
      <td>solo-fpp</td>
      <td>97</td>
      <td>95</td>
      <td>1560</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>49.75</td>
      <td>2</td>
      <td>0</td>
      <td>0.1875</td>
    </tr>
  </tbody>
</table>
</div>




```python
#I want to see the relationship between maxPlace, the true ranking from 1-100, with damageDealt.
#In other words, do you get better ranking if you deal more damage, and logically, by getting into fights?
data = []
for asset in np.random.choice(PG_train['Id'].unique(), 200):
    asset_df = PG_train[(PG_train['Id'] == asset)]

    data.append(go.Scatter(
        x = asset_df['damageDealt'].values,
        y = asset_df['maxPlace'].values,
        name = asset,
        showlegend=False
    ))
layout = go.Layout(dict(title = "Killing is Winning?",
                  xaxis = dict(title = 'Damage (Higher is Better)'),
                  yaxis = dict(title = 'Rank (Lower is Better)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
```


<div id="ce5ca402-22ce-4eb5-9fbd-4f99733bc3c6" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("ce5ca402-22ce-4eb5-9fbd-4f99733bc3c6", [{"name": "a1918013ce94aa", "showlegend": false, "x": [0.0], "y": [98], "type": "scatter", "uid": "2acd27d4-deca-11e8-ba7c-0088653b14b4"}, {"name": "002a8b3166c7d8", "showlegend": false, "x": [200.0], "y": [26], "type": "scatter", "uid": "2acd29f0-deca-11e8-be6f-0088653b14b4"}, {"name": "140880912aabc5", "showlegend": false, "x": [100.0], "y": [32], "type": "scatter", "uid": "2acd2b08-deca-11e8-b979-0088653b14b4"}, {"name": "11f610f79cf83f", "showlegend": false, "x": [89.45], "y": [26], "type": "scatter", "uid": "2acd2bf8-deca-11e8-8b7b-0088653b14b4"}, {"name": "6e2b4e1451dfad", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "2acd2ce8-deca-11e8-af27-0088653b14b4"}, {"name": "2906d89f211199", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acd2dd8-deca-11e8-b697-0088653b14b4"}, {"name": "46bbf9915dafeb", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "2acd2f0c-deca-11e8-8793-0088653b14b4"}, {"name": "a643749e79394f", "showlegend": false, "x": [100.0], "y": [49], "type": "scatter", "uid": "2acd2ffe-deca-11e8-8ceb-0088653b14b4"}, {"name": "71d807a7fa4b23", "showlegend": false, "x": [0.0], "y": [94], "type": "scatter", "uid": "2acd30ee-deca-11e8-9c07-0088653b14b4"}, {"name": "16a426fb4ff496", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "2acd31de-deca-11e8-81ae-0088653b14b4"}, {"name": "f4335357bdeb2a", "showlegend": false, "x": [23.22], "y": [47], "type": "scatter", "uid": "2acd32c6-deca-11e8-a697-0088653b14b4"}, {"name": "3a3c110aa1aea3", "showlegend": false, "x": [221.9], "y": [49], "type": "scatter", "uid": "2acd33a8-deca-11e8-b7e3-0088653b14b4"}, {"name": "866955b0474a2b", "showlegend": false, "x": [75.56], "y": [49], "type": "scatter", "uid": "2acd34d8-deca-11e8-8410-0088653b14b4"}, {"name": "58f3578ec13b56", "showlegend": false, "x": [184.6], "y": [28], "type": "scatter", "uid": "2acd35d0-deca-11e8-ab3d-0088653b14b4"}, {"name": "e0a149e15ec3c6", "showlegend": false, "x": [0.0], "y": [24], "type": "scatter", "uid": "2acd36b6-deca-11e8-aa8e-0088653b14b4"}, {"name": "14333ed0cfbfbd", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acd379c-deca-11e8-8085-0088653b14b4"}, {"name": "da6970ddeff9b4", "showlegend": false, "x": [69.0], "y": [49], "type": "scatter", "uid": "2acd3882-deca-11e8-8802-0088653b14b4"}, {"name": "fa3238ed2da528", "showlegend": false, "x": [598.5], "y": [26], "type": "scatter", "uid": "2acd3968-deca-11e8-b79d-0088653b14b4"}, {"name": "3ec03fcc986858", "showlegend": false, "x": [368.7], "y": [97], "type": "scatter", "uid": "2acd3a58-deca-11e8-abd7-0088653b14b4"}, {"name": "2db813a4ef3827", "showlegend": false, "x": [557.1], "y": [49], "type": "scatter", "uid": "2acd3b3e-deca-11e8-be8e-0088653b14b4"}, {"name": "cfd0a6b97ce2aa", "showlegend": false, "x": [31.68], "y": [28], "type": "scatter", "uid": "2acd3c24-deca-11e8-8fc1-0088653b14b4"}, {"name": "82607fe9ac5986", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "2acd3d0a-deca-11e8-bae1-0088653b14b4"}, {"name": "8fbf3e5b3fc438", "showlegend": false, "x": [121.3], "y": [28], "type": "scatter", "uid": "2acd3df0-deca-11e8-894c-0088653b14b4"}, {"name": "1ee528341a4ad5", "showlegend": false, "x": [350.5], "y": [48], "type": "scatter", "uid": "2acd3ed8-deca-11e8-8762-0088653b14b4"}, {"name": "fdf47fcfe727f0", "showlegend": false, "x": [791.7], "y": [8], "type": "scatter", "uid": "2acd3fba-deca-11e8-98a2-0088653b14b4"}, {"name": "ec9404b06496a5", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "2acd40e8-deca-11e8-9bb5-0088653b14b4"}, {"name": "d40b564753d577", "showlegend": false, "x": [183.7], "y": [97], "type": "scatter", "uid": "2acd41d8-deca-11e8-a0cb-0088653b14b4"}, {"name": "34a9b52a98f0cf", "showlegend": false, "x": [49.0], "y": [99], "type": "scatter", "uid": "2acd42b4-deca-11e8-a45a-0088653b14b4"}, {"name": "8765c80a361ecc", "showlegend": false, "x": [18.38], "y": [29], "type": "scatter", "uid": "2acd439a-deca-11e8-80df-0088653b14b4"}, {"name": "99e27e89c44372", "showlegend": false, "x": [177.7], "y": [94], "type": "scatter", "uid": "2acd448a-deca-11e8-a440-0088653b14b4"}, {"name": "f71287a7ae3940", "showlegend": false, "x": [769.5], "y": [48], "type": "scatter", "uid": "2acd4570-deca-11e8-b962-0088653b14b4"}, {"name": "ea2168030b96c5", "showlegend": false, "x": [12.96], "y": [97], "type": "scatter", "uid": "2acd4658-deca-11e8-b314-0088653b14b4"}, {"name": "741f2fbb07a6bb", "showlegend": false, "x": [300.0], "y": [28], "type": "scatter", "uid": "2acd473a-deca-11e8-b59f-0088653b14b4"}, {"name": "045476e29551a2", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "2acd4822-deca-11e8-b390-0088653b14b4"}, {"name": "63c59915e61dcb", "showlegend": false, "x": [341.4], "y": [48], "type": "scatter", "uid": "2acd4908-deca-11e8-b21b-0088653b14b4"}, {"name": "21fc4704aa44c1", "showlegend": false, "x": [19.35], "y": [26], "type": "scatter", "uid": "2acd49ee-deca-11e8-bbc2-0088653b14b4"}, {"name": "35b643c92c8ff8", "showlegend": false, "x": [171.9], "y": [25], "type": "scatter", "uid": "2acd4ad4-deca-11e8-9d2f-0088653b14b4"}, {"name": "be1465e5627ded", "showlegend": false, "x": [292.9], "y": [29], "type": "scatter", "uid": "2acd4bba-deca-11e8-a08c-0088653b14b4"}, {"name": "ce88092dfbb0f1", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "2acd4d2c-deca-11e8-a8ab-0088653b14b4"}, {"name": "505b454050c92f", "showlegend": false, "x": [73.13], "y": [50], "type": "scatter", "uid": "2acd4e1c-deca-11e8-8b61-0088653b14b4"}, {"name": "c9c8f56e1231c5", "showlegend": false, "x": [100.0], "y": [48], "type": "scatter", "uid": "2acd4f02-deca-11e8-8896-0088653b14b4"}, {"name": "6436ab5d189492", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acd4fe8-deca-11e8-843b-0088653b14b4"}, {"name": "738661ffc5052d", "showlegend": false, "x": [200.0], "y": [48], "type": "scatter", "uid": "2acd50cc-deca-11e8-8b0b-0088653b14b4"}, {"name": "eb5f531ee01e2f", "showlegend": false, "x": [109.7], "y": [94], "type": "scatter", "uid": "2acd51b4-deca-11e8-88ab-0088653b14b4"}, {"name": "9a327e020dbbe4", "showlegend": false, "x": [426.5], "y": [27], "type": "scatter", "uid": "2acd52a4-deca-11e8-bf57-0088653b14b4"}, {"name": "72395fc6445d4c", "showlegend": false, "x": [73.26], "y": [29], "type": "scatter", "uid": "2acd538a-deca-11e8-bc8b-0088653b14b4"}, {"name": "9ca3583a273698", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acd5470-deca-11e8-830b-0088653b14b4"}, {"name": "5786023827ac51", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acd5558-deca-11e8-85fc-0088653b14b4"}, {"name": "349401767a0be7", "showlegend": false, "x": [31.68], "y": [26], "type": "scatter", "uid": "2acd563a-deca-11e8-b8a0-0088653b14b4"}, {"name": "6650f974aa1779", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "2acd5722-deca-11e8-853c-0088653b14b4"}, {"name": "0f268ffe3ff469", "showlegend": false, "x": [100.0], "y": [30], "type": "scatter", "uid": "2acd5812-deca-11e8-940f-0088653b14b4"}, {"name": "d9a87d3940c0c0", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "2acd593e-deca-11e8-be75-0088653b14b4"}, {"name": "851dc7ab17aeb6", "showlegend": false, "x": [200.0], "y": [48], "type": "scatter", "uid": "2acd5a68-deca-11e8-a3a7-0088653b14b4"}, {"name": "155ad58a26be74", "showlegend": false, "x": [61.45], "y": [27], "type": "scatter", "uid": "2acd5b5a-deca-11e8-9ef2-0088653b14b4"}, {"name": "30c74cfdc30d77", "showlegend": false, "x": [111.3], "y": [27], "type": "scatter", "uid": "2acd5c4a-deca-11e8-878a-0088653b14b4"}, {"name": "4af36562121061", "showlegend": false, "x": [51.85], "y": [90], "type": "scatter", "uid": "2acd5d26-deca-11e8-93f0-0088653b14b4"}, {"name": "ef7881d1c7e09b", "showlegend": false, "x": [88.3], "y": [49], "type": "scatter", "uid": "2acd5e0c-deca-11e8-8113-0088653b14b4"}, {"name": "515bf41b3a0812", "showlegend": false, "x": [174.1], "y": [28], "type": "scatter", "uid": "2acd5ef4-deca-11e8-95ba-0088653b14b4"}, {"name": "4a750c157a4b47", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "2acd5fd8-deca-11e8-9481-0088653b14b4"}, {"name": "ae8f4d1175bb6a", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "2acd60be-deca-11e8-b08e-0088653b14b4"}, {"name": "a23f92d28d2435", "showlegend": false, "x": [95.2], "y": [26], "type": "scatter", "uid": "2acd61a4-deca-11e8-8984-0088653b14b4"}, {"name": "b24a5a20d5a46d", "showlegend": false, "x": [100.0], "y": [96], "type": "scatter", "uid": "2acd628a-deca-11e8-9ea5-0088653b14b4"}, {"name": "f94bd31f199012", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "2acd6370-deca-11e8-9ace-0088653b14b4"}, {"name": "8fb7249e6ad14c", "showlegend": false, "x": [1100.0], "y": [32], "type": "scatter", "uid": "2acd6458-deca-11e8-9478-0088653b14b4"}, {"name": "1ba247ee098762", "showlegend": false, "x": [49.0], "y": [28], "type": "scatter", "uid": "2acd6582-deca-11e8-9be8-0088653b14b4"}, {"name": "d6ac81e2bae199", "showlegend": false, "x": [121.6], "y": [29], "type": "scatter", "uid": "2acd6674-deca-11e8-9959-0088653b14b4"}, {"name": "b2fe46d960af3d", "showlegend": false, "x": [186.7], "y": [96], "type": "scatter", "uid": "2acd6758-deca-11e8-b1ef-0088653b14b4"}, {"name": "8acf8fac9a6f66", "showlegend": false, "x": [30.03], "y": [98], "type": "scatter", "uid": "2acd683e-deca-11e8-81eb-0088653b14b4"}, {"name": "c04ff439b4d1b7", "showlegend": false, "x": [107.9], "y": [28], "type": "scatter", "uid": "2acd6924-deca-11e8-9910-0088653b14b4"}, {"name": "4c147ff534e64c", "showlegend": false, "x": [532.1], "y": [50], "type": "scatter", "uid": "2acd6a0a-deca-11e8-b78a-0088653b14b4"}, {"name": "1c9ba3cd846a25", "showlegend": false, "x": [100.0], "y": [29], "type": "scatter", "uid": "2acd6af0-deca-11e8-8a6f-0088653b14b4"}, {"name": "40bd1cae635c25", "showlegend": false, "x": [101.0], "y": [25], "type": "scatter", "uid": "2acd6bd8-deca-11e8-bef1-0088653b14b4"}, {"name": "52e6d53d3bce40", "showlegend": false, "x": [36.72], "y": [48], "type": "scatter", "uid": "2acd6cba-deca-11e8-bf3c-0088653b14b4"}, {"name": "ae25b36709a09c", "showlegend": false, "x": [155.5], "y": [48], "type": "scatter", "uid": "2acd6da2-deca-11e8-b28e-0088653b14b4"}, {"name": "42e3d80f285c6b", "showlegend": false, "x": [119.3], "y": [46], "type": "scatter", "uid": "2acd6e88-deca-11e8-83e1-0088653b14b4"}, {"name": "190227c58e67bf", "showlegend": false, "x": [417.7], "y": [48], "type": "scatter", "uid": "2acd6f6e-deca-11e8-ba89-0088653b14b4"}, {"name": "096f620dfd098c", "showlegend": false, "x": [200.0], "y": [47], "type": "scatter", "uid": "2acd7054-deca-11e8-84c4-0088653b14b4"}, {"name": "e777b3d5901d98", "showlegend": false, "x": [100.0], "y": [95], "type": "scatter", "uid": "2acd7180-deca-11e8-aaa4-0088653b14b4"}, {"name": "14a2515a158714", "showlegend": false, "x": [139.2], "y": [27], "type": "scatter", "uid": "2acd72c0-deca-11e8-97fa-0088653b14b4"}, {"name": "889d077b1b59a2", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "2acd73a6-deca-11e8-a201-0088653b14b4"}, {"name": "03786150e5bcd8", "showlegend": false, "x": [46.24], "y": [47], "type": "scatter", "uid": "2acd748c-deca-11e8-9512-0088653b14b4"}, {"name": "914e28f6d1dc5b", "showlegend": false, "x": [135.3], "y": [28], "type": "scatter", "uid": "2acd7574-deca-11e8-bc22-0088653b14b4"}, {"name": "217c095ecda9f5", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "2acd7658-deca-11e8-8bcc-0088653b14b4"}, {"name": "6525ce513000b9", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acd773e-deca-11e8-9e02-0088653b14b4"}, {"name": "05c262004114c8", "showlegend": false, "x": [195.9], "y": [97], "type": "scatter", "uid": "2acd7824-deca-11e8-a618-0088653b14b4"}, {"name": "3f8e47df49dd78", "showlegend": false, "x": [312.2], "y": [29], "type": "scatter", "uid": "2acd790a-deca-11e8-88d0-0088653b14b4"}, {"name": "3dba5786287cd6", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "2acd79f0-deca-11e8-b249-0088653b14b4"}, {"name": "eb81a993de4901", "showlegend": false, "x": [275.6], "y": [30], "type": "scatter", "uid": "2acd7ad8-deca-11e8-beb8-0088653b14b4"}, {"name": "dff09b5197d7fb", "showlegend": false, "x": [438.5], "y": [46], "type": "scatter", "uid": "2acd7bba-deca-11e8-a9c2-0088653b14b4"}, {"name": "05d644c79367fa", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acd7ca2-deca-11e8-bf6c-0088653b14b4"}, {"name": "dd088fa7b504d9", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acd7dcc-deca-11e8-be1f-0088653b14b4"}, {"name": "9ca1d3a9bbad6e", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "2acd7ebe-deca-11e8-9cff-0088653b14b4"}, {"name": "b829cc65e6c8a8", "showlegend": false, "x": [420.4], "y": [98], "type": "scatter", "uid": "2acd7fa4-deca-11e8-a6c3-0088653b14b4"}, {"name": "ddf7771843a282", "showlegend": false, "x": [30.06], "y": [26], "type": "scatter", "uid": "2acd808a-deca-11e8-b31f-0088653b14b4"}, {"name": "f5d6a8c67153a6", "showlegend": false, "x": [119.6], "y": [48], "type": "scatter", "uid": "2acd8170-deca-11e8-9dbb-0088653b14b4"}, {"name": "9684a57053d171", "showlegend": false, "x": [25.86], "y": [26], "type": "scatter", "uid": "2acd8258-deca-11e8-badd-0088653b14b4"}, {"name": "fdb4cf3d1a6260", "showlegend": false, "x": [353.5], "y": [30], "type": "scatter", "uid": "2acd833a-deca-11e8-9d80-0088653b14b4"}, {"name": "3be8e096b1943f", "showlegend": false, "x": [180.3], "y": [48], "type": "scatter", "uid": "2acd8422-deca-11e8-a4a5-0088653b14b4"}, {"name": "59fb6d725d0130", "showlegend": false, "x": [55.65], "y": [47], "type": "scatter", "uid": "2acd854c-deca-11e8-86d7-0088653b14b4"}, {"name": "72da1a5bdd58f6", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "2acd863e-deca-11e8-8981-0088653b14b4"}, {"name": "d83f5ddc8309e1", "showlegend": false, "x": [25.93], "y": [30], "type": "scatter", "uid": "2acd8724-deca-11e8-87e1-0088653b14b4"}, {"name": "480685b84498d3", "showlegend": false, "x": [390.6], "y": [97], "type": "scatter", "uid": "2acd880a-deca-11e8-ab8d-0088653b14b4"}, {"name": "53eb2e20a2e3f2", "showlegend": false, "x": [169.0], "y": [29], "type": "scatter", "uid": "2acd88f0-deca-11e8-8caf-0088653b14b4"}, {"name": "053b2eb7b55942", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "2acd8a1c-deca-11e8-a944-0088653b14b4"}, {"name": "3cb638ca14204d", "showlegend": false, "x": [170.4], "y": [95], "type": "scatter", "uid": "2acd8b0c-deca-11e8-8090-0088653b14b4"}, {"name": "cc22371aaf6801", "showlegend": false, "x": [9.778], "y": [42], "type": "scatter", "uid": "2acd8bf4-deca-11e8-ac19-0088653b14b4"}, {"name": "e4056bb8383826", "showlegend": false, "x": [23.05], "y": [46], "type": "scatter", "uid": "2acd8cd8-deca-11e8-8300-0088653b14b4"}, {"name": "936681b418c795", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "2acd8dbe-deca-11e8-b1f0-0088653b14b4"}, {"name": "44e7dac5284fc9", "showlegend": false, "x": [112.1], "y": [47], "type": "scatter", "uid": "2acd8ea4-deca-11e8-908e-0088653b14b4"}, {"name": "bcc134e2b7610b", "showlegend": false, "x": [279.9], "y": [28], "type": "scatter", "uid": "2acd8f8a-deca-11e8-8fb5-0088653b14b4"}, {"name": "efad32973ab70a", "showlegend": false, "x": [517.9], "y": [49], "type": "scatter", "uid": "2acd9070-deca-11e8-8a9f-0088653b14b4"}, {"name": "4110071356b2d2", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acd9158-deca-11e8-bd1c-0088653b14b4"}, {"name": "826bb44de1c6b7", "showlegend": false, "x": [127.7], "y": [47], "type": "scatter", "uid": "2acd923a-deca-11e8-aa1e-0088653b14b4"}, {"name": "2dce20dc9a2878", "showlegend": false, "x": [250.7], "y": [50], "type": "scatter", "uid": "2acd9322-deca-11e8-a421-0088653b14b4"}, {"name": "0eea84a271055c", "showlegend": false, "x": [0.0], "y": [89], "type": "scatter", "uid": "2acd9408-deca-11e8-8d0f-0088653b14b4"}, {"name": "56b40d8de97220", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acd94ee-deca-11e8-b063-0088653b14b4"}, {"name": "ffd3d4abea7d9a", "showlegend": false, "x": [54.18], "y": [47], "type": "scatter", "uid": "2acd9624-deca-11e8-9821-0088653b14b4"}, {"name": "9bb7245107c162", "showlegend": false, "x": [354.1], "y": [28], "type": "scatter", "uid": "2acd9714-deca-11e8-b5f9-0088653b14b4"}, {"name": "2a24089647b3a4", "showlegend": false, "x": [300.0], "y": [27], "type": "scatter", "uid": "2acd97fa-deca-11e8-a385-0088653b14b4"}, {"name": "976060d302b8d2", "showlegend": false, "x": [33.11], "y": [26], "type": "scatter", "uid": "2acd98e2-deca-11e8-8040-0088653b14b4"}, {"name": "0fd8e8c6bfce28", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "2acd99c6-deca-11e8-99c5-0088653b14b4"}, {"name": "8d06a63ac9bfb9", "showlegend": false, "x": [115.8], "y": [96], "type": "scatter", "uid": "2acd9aac-deca-11e8-a839-0088653b14b4"}, {"name": "d57de9dae56aeb", "showlegend": false, "x": [27.48], "y": [97], "type": "scatter", "uid": "2acd9b88-deca-11e8-aaf2-0088653b14b4"}, {"name": "7ff814b141706c", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acd9c6e-deca-11e8-8c7c-0088653b14b4"}, {"name": "81b5b1f15f9936", "showlegend": false, "x": [18.94], "y": [29], "type": "scatter", "uid": "2acd9d5e-deca-11e8-ac2b-0088653b14b4"}, {"name": "e2f353f8922a88", "showlegend": false, "x": [159.8], "y": [27], "type": "scatter", "uid": "2acd9e46-deca-11e8-9e4e-0088653b14b4"}, {"name": "0abbe6e81996a8", "showlegend": false, "x": [100.0], "y": [29], "type": "scatter", "uid": "2acd9f28-deca-11e8-9bc6-0088653b14b4"}, {"name": "20b79a5b0463d7", "showlegend": false, "x": [270.2], "y": [26], "type": "scatter", "uid": "2acda010-deca-11e8-a81a-0088653b14b4"}, {"name": "f485c54e3c9652", "showlegend": false, "x": [0.0], "y": [25], "type": "scatter", "uid": "2acda0f6-deca-11e8-b97d-0088653b14b4"}, {"name": "a18b77447b581a", "showlegend": false, "x": [22.45], "y": [98], "type": "scatter", "uid": "2acda218-deca-11e8-bfb6-0088653b14b4"}, {"name": "1d3d84e9e63b81", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acda308-deca-11e8-a6e2-0088653b14b4"}, {"name": "23cf196de039b0", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "2acda3ee-deca-11e8-b014-0088653b14b4"}, {"name": "07304c4af4a7e1", "showlegend": false, "x": [322.0], "y": [27], "type": "scatter", "uid": "2acda4d4-deca-11e8-84b9-0088653b14b4"}, {"name": "6868c2ee62a814", "showlegend": false, "x": [166.2], "y": [48], "type": "scatter", "uid": "2acda5ba-deca-11e8-90ec-0088653b14b4"}, {"name": "38595cae1e8191", "showlegend": false, "x": [298.0], "y": [48], "type": "scatter", "uid": "2acda69e-deca-11e8-87af-0088653b14b4"}, {"name": "6d606536793f69", "showlegend": false, "x": [153.9], "y": [28], "type": "scatter", "uid": "2acda786-deca-11e8-a9cb-0088653b14b4"}, {"name": "362cc6edb30886", "showlegend": false, "x": [2.41], "y": [28], "type": "scatter", "uid": "2acda86c-deca-11e8-8a33-0088653b14b4"}, {"name": "0a88f7baebeb59", "showlegend": false, "x": [218.5], "y": [30], "type": "scatter", "uid": "2acda95c-deca-11e8-a5a3-0088653b14b4"}, {"name": "eb9f86c89d2060", "showlegend": false, "x": [141.0], "y": [28], "type": "scatter", "uid": "2acdaa42-deca-11e8-a6a5-0088653b14b4"}, {"name": "5558bc0b2287fd", "showlegend": false, "x": [127.5], "y": [45], "type": "scatter", "uid": "2acdab28-deca-11e8-a43c-0088653b14b4"}, {"name": "c5ad2da04fb65a", "showlegend": false, "x": [100.0], "y": [27], "type": "scatter", "uid": "2acdac0c-deca-11e8-8ed5-0088653b14b4"}, {"name": "6c428fc4b67d72", "showlegend": false, "x": [100.0], "y": [31], "type": "scatter", "uid": "2acdacf4-deca-11e8-9b30-0088653b14b4"}, {"name": "ca23ef5e3b189f", "showlegend": false, "x": [171.0], "y": [50], "type": "scatter", "uid": "2acdae18-deca-11e8-92f2-0088653b14b4"}, {"name": "2b386381f12115", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "2acdaf06-deca-11e8-a1d9-0088653b14b4"}, {"name": "5e104cd8e188fe", "showlegend": false, "x": [95.26], "y": [47], "type": "scatter", "uid": "2acdb028-deca-11e8-ab44-0088653b14b4"}, {"name": "8404900b1343db", "showlegend": false, "x": [70.2], "y": [27], "type": "scatter", "uid": "2acdb118-deca-11e8-bc3d-0088653b14b4"}, {"name": "7b93026909abf3", "showlegend": false, "x": [279.1], "y": [49], "type": "scatter", "uid": "2acdb1fe-deca-11e8-b7bd-0088653b14b4"}, {"name": "c60c7ea86cac63", "showlegend": false, "x": [100.0], "y": [31], "type": "scatter", "uid": "2acdb2e4-deca-11e8-90ee-0088653b14b4"}, {"name": "bedba87e87da71", "showlegend": false, "x": [0.0], "y": [96], "type": "scatter", "uid": "2acdb3ca-deca-11e8-8717-0088653b14b4"}, {"name": "a23953c0d9b6a6", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "2acdb4b0-deca-11e8-aea2-0088653b14b4"}, {"name": "454940293f37a8", "showlegend": false, "x": [23.22], "y": [28], "type": "scatter", "uid": "2acdb598-deca-11e8-a677-0088653b14b4"}, {"name": "092cb2abd4c8d2", "showlegend": false, "x": [119.9], "y": [29], "type": "scatter", "uid": "2acdb67a-deca-11e8-b957-0088653b14b4"}, {"name": "ad53fbff89d3df", "showlegend": false, "x": [29.06], "y": [26], "type": "scatter", "uid": "2acdb762-deca-11e8-937b-0088653b14b4"}, {"name": "129bf74e5d95ae", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acdb848-deca-11e8-92b9-0088653b14b4"}, {"name": "96a29f7771ac32", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acdb92e-deca-11e8-b521-0088653b14b4"}, {"name": "e31f6c87693d6c", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "2acdba5a-deca-11e8-a184-0088653b14b4"}, {"name": "79e8e1785f5209", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acdbb4a-deca-11e8-8dcf-0088653b14b4"}, {"name": "3f6fbf72d9053e", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "2acdbc30-deca-11e8-98a5-0088653b14b4"}, {"name": "d938d767c1a01c", "showlegend": false, "x": [436.0], "y": [27], "type": "scatter", "uid": "2acdbd18-deca-11e8-b41e-0088653b14b4"}, {"name": "4862591170280a", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "2acdbdfa-deca-11e8-a1cd-0088653b14b4"}, {"name": "e78f49b25b1cc9", "showlegend": false, "x": [320.4], "y": [46], "type": "scatter", "uid": "2acdbee2-deca-11e8-8d73-0088653b14b4"}, {"name": "53aeb9d1e8a814", "showlegend": false, "x": [93.6], "y": [98], "type": "scatter", "uid": "2acdbfc8-deca-11e8-b559-0088653b14b4"}, {"name": "a4ba447b222972", "showlegend": false, "x": [370.7], "y": [50], "type": "scatter", "uid": "2acdc0ae-deca-11e8-b86a-0088653b14b4"}, {"name": "d406f4dce89b51", "showlegend": false, "x": [170.7], "y": [29], "type": "scatter", "uid": "2acdc18a-deca-11e8-84ae-0088653b14b4"}, {"name": "a08d00f87a5abd", "showlegend": false, "x": [56.76], "y": [28], "type": "scatter", "uid": "2acdc270-deca-11e8-912c-0088653b14b4"}, {"name": "b64b49f2b7f4d1", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acdc358-deca-11e8-b0d0-0088653b14b4"}, {"name": "93ecba1006829c", "showlegend": false, "x": [154.3], "y": [28], "type": "scatter", "uid": "2acdc43a-deca-11e8-b917-0088653b14b4"}, {"name": "653c6eb3b3d753", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acdc522-deca-11e8-b873-0088653b14b4"}, {"name": "4e53a218cc8fc1", "showlegend": false, "x": [89.0], "y": [95], "type": "scatter", "uid": "2acdc64c-deca-11e8-9056-0088653b14b4"}, {"name": "69860ead6201ae", "showlegend": false, "x": [122.0], "y": [29], "type": "scatter", "uid": "2acdc73e-deca-11e8-9ece-0088653b14b4"}, {"name": "d1241fbdc912fb", "showlegend": false, "x": [200.0], "y": [50], "type": "scatter", "uid": "2acdc824-deca-11e8-8176-0088653b14b4"}, {"name": "4f5b5bd3b662ef", "showlegend": false, "x": [6.48], "y": [28], "type": "scatter", "uid": "2acdc90a-deca-11e8-a34c-0088653b14b4"}, {"name": "fe2510552addec", "showlegend": false, "x": [236.5], "y": [100], "type": "scatter", "uid": "2acdc9f0-deca-11e8-b179-0088653b14b4"}, {"name": "13db2f70a15c40", "showlegend": false, "x": [68.6], "y": [96], "type": "scatter", "uid": "2acdcad8-deca-11e8-b227-0088653b14b4"}, {"name": "ad328ac3ebe4d9", "showlegend": false, "x": [100.0], "y": [28], "type": "scatter", "uid": "2acdcbba-deca-11e8-8857-0088653b14b4"}, {"name": "3a90de6d7acef1", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "2acdcca2-deca-11e8-8b7f-0088653b14b4"}, {"name": "89230d55b35c80", "showlegend": false, "x": [100.0], "y": [47], "type": "scatter", "uid": "2acdcd92-deca-11e8-a21e-0088653b14b4"}, {"name": "3ca9ce8be89dbd", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acdce6e-deca-11e8-a0c2-0088653b14b4"}, {"name": "69d51b59bf4227", "showlegend": false, "x": [60.43], "y": [47], "type": "scatter", "uid": "2acdcf54-deca-11e8-9ffd-0088653b14b4"}, {"name": "9f7f2e8c1351c6", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "2acdd03a-deca-11e8-9c1b-0088653b14b4"}, {"name": "ab6ad5301b779e", "showlegend": false, "x": [104.8], "y": [30], "type": "scatter", "uid": "2acdd11e-deca-11e8-8c52-0088653b14b4"}, {"name": "1bce3a1bfd21a5", "showlegend": false, "x": [87.46], "y": [27], "type": "scatter", "uid": "2acdd24c-deca-11e8-8015-0088653b14b4"}, {"name": "43cd7c96840ca6", "showlegend": false, "x": [28.51], "y": [29], "type": "scatter", "uid": "2acdd334-deca-11e8-99a8-0088653b14b4"}, {"name": "b137b0a1de552b", "showlegend": false, "x": [31.66], "y": [16], "type": "scatter", "uid": "2acdd418-deca-11e8-ae9f-0088653b14b4"}, {"name": "fc0c80a4d13ea9", "showlegend": false, "x": [0.0], "y": [95], "type": "scatter", "uid": "2acdd4fe-deca-11e8-9373-0088653b14b4"}, {"name": "eef51f1fdb3eeb", "showlegend": false, "x": [124.2], "y": [28], "type": "scatter", "uid": "2acdd5da-deca-11e8-b2d0-0088653b14b4"}, {"name": "32ea96aa72f73b", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acdd6c0-deca-11e8-a504-0088653b14b4"}, {"name": "49455c2aca5033", "showlegend": false, "x": [221.7], "y": [28], "type": "scatter", "uid": "2acdd7ec-deca-11e8-ad11-0088653b14b4"}, {"name": "5c569eac630fb4", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "2acdd8dc-deca-11e8-b8c9-0088653b14b4"}, {"name": "7f31f01e792539", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "2acdd9c2-deca-11e8-83f4-0088653b14b4"}, {"name": "04a4df35032110", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "2acddae4-deca-11e8-aef9-0088653b14b4"}, {"name": "36509e5b755688", "showlegend": false, "x": [857.1], "y": [28], "type": "scatter", "uid": "2acddbd4-deca-11e8-928a-0088653b14b4"}, {"name": "996c71968f5542", "showlegend": false, "x": [205.2], "y": [26], "type": "scatter", "uid": "2acddcb0-deca-11e8-940c-0088653b14b4"}, {"name": "4e4ad87476f568", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "2acddd98-deca-11e8-a9e6-0088653b14b4"}, {"name": "b24c20c3ee303d", "showlegend": false, "x": [0.0], "y": [23], "type": "scatter", "uid": "2acddec2-deca-11e8-8cf7-0088653b14b4"}, {"name": "e3b9069438641f", "showlegend": false, "x": [55.86], "y": [43], "type": "scatter", "uid": "2acddfb4-deca-11e8-b7b5-0088653b14b4"}, {"name": "4008a45ed584f8", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "2acde08c-deca-11e8-9b8c-0088653b14b4"}, {"name": "a5be95f02ad360", "showlegend": false, "x": [218.5], "y": [17], "type": "scatter", "uid": "2acde174-deca-11e8-9a5b-0088653b14b4"}, {"name": "cb11a02853da08", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "2acde250-deca-11e8-b720-0088653b14b4"}, {"name": "6601ccee87f0b8", "showlegend": false, "x": [40.62], "y": [29], "type": "scatter", "uid": "2acde322-deca-11e8-9626-0088653b14b4"}], {"legend": {"orientation": "h"}, "title": "Killing is Winning?", "xaxis": {"title": "Damage (Higher is Better)"}, "yaxis": {"title": "Rank (Lower is Better)"}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
PG_train2=PG_train[PG_train.damageDealt==0]
```


```python
PG_train2.head()
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
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>killStreaks</th>
      <th>longestKill</th>
      <th>matchDuration</th>
      <th>matchType</th>
      <th>maxPlace</th>
      <th>numGroups</th>
      <th>rankPoints</th>
      <th>revives</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7f96b2f878858a</td>
      <td>4d4b580de459be</td>
      <td>a10357fd1a4a91</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>1241</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1306</td>
      <td>squad-fpp</td>
      <td>28</td>
      <td>26</td>
      <td>-1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>244.8</td>
      <td>1</td>
      <td>1466</td>
      <td>0.4444</td>
    </tr>
    <tr>
      <th>6</th>
      <td>95959be0e21ca3</td>
      <td>2c485a1ad3d0f1</td>
      <td>a8274e903927a2</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1316</td>
      <td>squad-fpp</td>
      <td>28</td>
      <td>28</td>
      <td>-1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.5</td>
      <td>1</td>
      <td>1497</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ce4f6ac165705e</td>
      <td>da24cdb91969cc</td>
      <td>535b5dbd965a94</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1774</td>
      <td>squad-fpp</td>
      <td>29</td>
      <td>28</td>
      <td>1766</td>
      <td>0</td>
      <td>6639.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2784.0</td>
      <td>6</td>
      <td>0</td>
      <td>0.9286</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7bd224781f064b</td>
      <td>6dde607d151819</td>
      <td>733af30cc00099</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1301</td>
      <td>squad-fpp</td>
      <td>27</td>
      <td>26</td>
      <td>1355</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>137.4</td>
      <td>2</td>
      <td>0</td>
      <td>0.1923</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ac5b57ff39979c</td>
      <td>857cc55b2b6001</td>
      <td>e019e04dee4f19</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>87</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1530</td>
      <td>duo</td>
      <td>46</td>
      <td>44</td>
      <td>1534</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#I want to see the relationship between maxPlace, the true ranking from 1-100, when damageDealt is null.
#In other words, what's your ranking if you deal no damage, and logically, by avoiding getting into fights?
data = []
for asset in np.random.choice(PG_train2['Id'].unique(), 200):
    asset_df = PG_train2[(PG_train2['Id'] == asset)]

    data.append(go.Scatter(
        x = asset_df['damageDealt'].values,
        y = asset_df['maxPlace'].values,
        name = asset,
        showlegend=False
    ))
layout = go.Layout(dict(title = "No Killing is Winning?",
                  xaxis = dict(title = 'No Damage'),
                  yaxis = dict(title = 'Rank (Lower is Better)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
```


<div id="375742c1-dd26-4139-acb0-1dc5ba8b1d56" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("375742c1-dd26-4139-acb0-1dc5ba8b1d56", [{"name": "e3510286f5f527", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d00055c-deca-11e8-a122-0088653b14b4"}, {"name": "7e3f0e3e247f97", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d0007be-deca-11e8-9d31-0088653b14b4"}, {"name": "c3a967a510afd2", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d000962-deca-11e8-ba27-0088653b14b4"}, {"name": "2c84d806030234", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d000a66-deca-11e8-ad7c-0088653b14b4"}, {"name": "5103f7567970d4", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d000b62-deca-11e8-aa81-0088653b14b4"}, {"name": "74e29bff92a508", "showlegend": false, "x": [0.0], "y": [89], "type": "scatter", "uid": "3d000c5a-deca-11e8-9071-0088653b14b4"}, {"name": "6dabadfbd69fa6", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d000da4-deca-11e8-8d0a-0088653b14b4"}, {"name": "0dde92ab32d6f9", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d000e8a-deca-11e8-9876-0088653b14b4"}, {"name": "fc7936dd08b847", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d001010-deca-11e8-8ce0-0088653b14b4"}, {"name": "8d061fedc2c4b9", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d001128-deca-11e8-988e-0088653b14b4"}, {"name": "0f183285931e56", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d001222-deca-11e8-8605-0088653b14b4"}, {"name": "ba918e29dec39e", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d00131c-deca-11e8-9ff6-0088653b14b4"}, {"name": "48d66069a1751c", "showlegend": false, "x": [0.0], "y": [98], "type": "scatter", "uid": "3d001418-deca-11e8-8eab-0088653b14b4"}, {"name": "ab78e72fb804ff", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d001506-deca-11e8-a1ee-0088653b14b4"}, {"name": "5a14877094704a", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d001600-deca-11e8-88e9-0088653b14b4"}, {"name": "d7bcd0d2f1be17", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d0016f0-deca-11e8-8313-0088653b14b4"}, {"name": "b07b5caab3d60d", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d0017e8-deca-11e8-92ba-0088653b14b4"}, {"name": "44aab80fee51d5", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d0018da-deca-11e8-b718-0088653b14b4"}, {"name": "5bafb266098dd0", "showlegend": false, "x": [0.0], "y": [91], "type": "scatter", "uid": "3d0019ca-deca-11e8-be5b-0088653b14b4"}, {"name": "5519fa87dc7e62", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d001ac6-deca-11e8-9681-0088653b14b4"}, {"name": "5f92bd670f6351", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d001d3a-deca-11e8-9944-0088653b14b4"}, {"name": "c34b05ec5d1bc3", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d002034-deca-11e8-a21d-0088653b14b4"}, {"name": "d074c8facb7628", "showlegend": false, "x": [0.0], "y": [93], "type": "scatter", "uid": "3d0022da-deca-11e8-8ac1-0088653b14b4"}, {"name": "9165e24f6903ca", "showlegend": false, "x": [0.0], "y": [99], "type": "scatter", "uid": "3d00241a-deca-11e8-bd6d-0088653b14b4"}, {"name": "4292d571ae7eee", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d002514-deca-11e8-b5fe-0088653b14b4"}, {"name": "8b3534014df461", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d002686-deca-11e8-a6b4-0088653b14b4"}, {"name": "2c824c02f59680", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d002780-deca-11e8-b4b5-0088653b14b4"}, {"name": "48d34226257a79", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00287a-deca-11e8-be74-0088653b14b4"}, {"name": "b54ad824df8dc4", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d002974-deca-11e8-982b-0088653b14b4"}, {"name": "fbc8eab77dc947", "showlegend": false, "x": [0.0], "y": [95], "type": "scatter", "uid": "3d002a64-deca-11e8-ba74-0088653b14b4"}, {"name": "ca922cbda3fe67", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d002b54-deca-11e8-9eec-0088653b14b4"}, {"name": "32513e2191784c", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d002c46-deca-11e8-93c5-0088653b14b4"}, {"name": "1d892407104871", "showlegend": false, "x": [0.0], "y": [43], "type": "scatter", "uid": "3d002d3e-deca-11e8-adff-0088653b14b4"}, {"name": "92ed15cbf169a1", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d002e2e-deca-11e8-be0a-0088653b14b4"}, {"name": "ed9f0dada9d47e", "showlegend": false, "x": [0.0], "y": [98], "type": "scatter", "uid": "3d002f5a-deca-11e8-9bab-0088653b14b4"}, {"name": "87bc7d1029e392", "showlegend": false, "x": [0.0], "y": [25], "type": "scatter", "uid": "3d003074-deca-11e8-b8a1-0088653b14b4"}, {"name": "aaea668b55e05a", "showlegend": false, "x": [0.0], "y": [95], "type": "scatter", "uid": "3d003162-deca-11e8-a1a2-0088653b14b4"}, {"name": "e3a2705bec94e8", "showlegend": false, "x": [0.0], "y": [22], "type": "scatter", "uid": "3d00325c-deca-11e8-9041-0088653b14b4"}, {"name": "f76dbb15ae95a6", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d003358-deca-11e8-b189-0088653b14b4"}, {"name": "21a5ca08d87a70", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d003446-deca-11e8-8005-0088653b14b4"}, {"name": "5305ea6101dc78", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d003540-deca-11e8-a360-0088653b14b4"}, {"name": "43194f35903dc1", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d003630-deca-11e8-8604-0088653b14b4"}, {"name": "d5ad7e73dea477", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "3d00371e-deca-11e8-a331-0088653b14b4"}, {"name": "2e8bb4973b1f20", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00381a-deca-11e8-b790-0088653b14b4"}, {"name": "3f9fddc6a00e20", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00395a-deca-11e8-8744-0088653b14b4"}, {"name": "13f9f03eda6feb", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d003a54-deca-11e8-bda0-0088653b14b4"}, {"name": "e22bf772f4ce98", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d003b4c-deca-11e8-a3ce-0088653b14b4"}, {"name": "8df1303e2bf7fe", "showlegend": false, "x": [0.0], "y": [100], "type": "scatter", "uid": "3d003c7a-deca-11e8-89c5-0088653b14b4"}, {"name": "552bc43207f583", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d003d7e-deca-11e8-8bf2-0088653b14b4"}, {"name": "133b30bf17657e", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d004076-deca-11e8-86e4-0088653b14b4"}, {"name": "d9df8dbf362d03", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d004206-deca-11e8-8a10-0088653b14b4"}, {"name": "75c2a68adb626e", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d00451c-deca-11e8-8b2b-0088653b14b4"}, {"name": "dae6990717828c", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "3d004648-deca-11e8-8948-0088653b14b4"}, {"name": "acc4b1f3027c39", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d004738-deca-11e8-83b3-0088653b14b4"}, {"name": "00d5eca3062113", "showlegend": false, "x": [0.0], "y": [93], "type": "scatter", "uid": "3d004828-deca-11e8-9b12-0088653b14b4"}, {"name": "e08b3ff6cfb15e", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d004968-deca-11e8-adf1-0088653b14b4"}, {"name": "7c40757fc8b03d", "showlegend": false, "x": [0.0], "y": [45], "type": "scatter", "uid": "3d004ab4-deca-11e8-b7fd-0088653b14b4"}, {"name": "e3340273393abd", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d004c62-deca-11e8-9d8e-0088653b14b4"}, {"name": "aed589e2cf4006", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d004ec2-deca-11e8-9b7e-0088653b14b4"}, {"name": "317466faf0db25", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d004fee-deca-11e8-8721-0088653b14b4"}, {"name": "36843c392a00b8", "showlegend": false, "x": [0.0], "y": [91], "type": "scatter", "uid": "3d005158-deca-11e8-8345-0088653b14b4"}, {"name": "e5e39ae871af0d", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d005278-deca-11e8-92da-0088653b14b4"}, {"name": "2bd55c4775afc8", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00537a-deca-11e8-94b3-0088653b14b4"}, {"name": "e5d976e1fe3122", "showlegend": false, "x": [0.0], "y": [45], "type": "scatter", "uid": "3d005480-deca-11e8-8f5a-0088653b14b4"}, {"name": "546b9754753094", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d00557a-deca-11e8-b90e-0088653b14b4"}, {"name": "9c76cd04140734", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d005674-deca-11e8-b652-0088653b14b4"}, {"name": "ab3fd3cfcd9a10", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00576e-deca-11e8-b256-0088653b14b4"}, {"name": "96a7c11572a624", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d005868-deca-11e8-8071-0088653b14b4"}, {"name": "0276bf5bf63730", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "3d005962-deca-11e8-96aa-0088653b14b4"}, {"name": "f071dfd88f09a8", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d005a5c-deca-11e8-b1c3-0088653b14b4"}, {"name": "503e331652a214", "showlegend": false, "x": [0.0], "y": [94], "type": "scatter", "uid": "3d005b4c-deca-11e8-b106-0088653b14b4"}, {"name": "cf5700eb30b94b", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d005c46-deca-11e8-9b52-0088653b14b4"}, {"name": "eaa5a6212bf8e6", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d005d40-deca-11e8-8696-0088653b14b4"}, {"name": "2b8f591ae0d860", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d005e80-deca-11e8-9b51-0088653b14b4"}, {"name": "b34f3b9f5bac7a", "showlegend": false, "x": [0.0], "y": [25], "type": "scatter", "uid": "3d005f8c-deca-11e8-a930-0088653b14b4"}, {"name": "88638b75799bd7", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d006088-deca-11e8-9c5b-0088653b14b4"}, {"name": "715303602c985b", "showlegend": false, "x": [0.0], "y": [98], "type": "scatter", "uid": "3d006182-deca-11e8-a24e-0088653b14b4"}, {"name": "5ab72ea05af3d1", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d006274-deca-11e8-b0ed-0088653b14b4"}, {"name": "35548b3aa99fff", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d00636c-deca-11e8-966e-0088653b14b4"}, {"name": "5e9b2c39a41c7a", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d006466-deca-11e8-a941-0088653b14b4"}, {"name": "9a3100b95da7bd", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d006562-deca-11e8-b94f-0088653b14b4"}, {"name": "7b04f59693a03e", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d006650-deca-11e8-9c61-0088653b14b4"}, {"name": "c4796b33d0ac7e", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00674a-deca-11e8-9a62-0088653b14b4"}, {"name": "f3eb55aef08934", "showlegend": false, "x": [0.0], "y": [96], "type": "scatter", "uid": "3d00683a-deca-11e8-aaa4-0088653b14b4"}, {"name": "a304e52934aec9", "showlegend": false, "x": [0.0], "y": [84], "type": "scatter", "uid": "3d006934-deca-11e8-b9ba-0088653b14b4"}, {"name": "09b3132a0dfc10", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d006a2e-deca-11e8-8a28-0088653b14b4"}, {"name": "a8cc71abc5ad25", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d006b5a-deca-11e8-b30c-0088653b14b4"}, {"name": "9502d89972f0f4", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d006c68-deca-11e8-b250-0088653b14b4"}, {"name": "5f32aeab3b0fa5", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d006d62-deca-11e8-9318-0088653b14b4"}, {"name": "dea275e1751c27", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d006e5c-deca-11e8-a097-0088653b14b4"}, {"name": "1e29ccd36b4d55", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d006f9c-deca-11e8-8ecb-0088653b14b4"}, {"name": "91c65fce931f5f", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d0070a8-deca-11e8-9ca3-0088653b14b4"}, {"name": "709d234179a39f", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00719a-deca-11e8-911b-0088653b14b4"}, {"name": "c1494ec9614357", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d007294-deca-11e8-8097-0088653b14b4"}, {"name": "1a7cb3233ecdb5", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00738c-deca-11e8-870f-0088653b14b4"}, {"name": "9b467ec0ebca6f", "showlegend": false, "x": [0.0], "y": [90], "type": "scatter", "uid": "3d007488-deca-11e8-90b9-0088653b14b4"}, {"name": "69789fe8c2f3c3", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "3d007578-deca-11e8-8d58-0088653b14b4"}, {"name": "7b7febfd87b4db", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d007674-deca-11e8-a7ce-0088653b14b4"}, {"name": "93a66106b4242b", "showlegend": false, "x": [0.0], "y": [98], "type": "scatter", "uid": "3d007762-deca-11e8-b744-0088653b14b4"}, {"name": "38e5e0875c893f", "showlegend": false, "x": [0.0], "y": [99], "type": "scatter", "uid": "3d007938-deca-11e8-90d1-0088653b14b4"}, {"name": "d38fa73d0dffe4", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d007b4a-deca-11e8-95c0-0088653b14b4"}, {"name": "39d0b95cb14e98", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d007d28-deca-11e8-b65b-0088653b14b4"}, {"name": "a058b7d0370adb", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "3d007ef6-deca-11e8-9332-0088653b14b4"}, {"name": "1dde2ea1c114e0", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d0080cc-deca-11e8-b961-0088653b14b4"}, {"name": "9e0df70369f677", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d0082a2-deca-11e8-b959-0088653b14b4"}, {"name": "f0da1b513893d4", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d008482-deca-11e8-8037-0088653b14b4"}, {"name": "40ec1bfb4e6e53", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d00866c-deca-11e8-9618-0088653b14b4"}, {"name": "1fca6f2d0d9f18", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d008862-deca-11e8-8e34-0088653b14b4"}, {"name": "a21bc7437e45b0", "showlegend": false, "x": [0.0], "y": [95], "type": "scatter", "uid": "3d008a68-deca-11e8-9c7d-0088653b14b4"}, {"name": "8b0f43471477b0", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d008c52-deca-11e8-a19c-0088653b14b4"}, {"name": "d13e12bf42cadd", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d008e50-deca-11e8-9de7-0088653b14b4"}, {"name": "6c2d564c452959", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d009030-deca-11e8-a0dc-0088653b14b4"}, {"name": "e470835dac9be9", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d009288-deca-11e8-bb4b-0088653b14b4"}, {"name": "d2c86c91d6e477", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d009486-deca-11e8-8f62-0088653b14b4"}, {"name": "62b24b6cdd4c32", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d0099c2-deca-11e8-890e-0088653b14b4"}, {"name": "e184b29e5698e8", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d009c10-deca-11e8-ae55-0088653b14b4"}, {"name": "9643d586b8ef61", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d00a85e-deca-11e8-a846-0088653b14b4"}, {"name": "5a68201b0a8c3f", "showlegend": false, "x": [0.0], "y": [31], "type": "scatter", "uid": "3d00aa8c-deca-11e8-b533-0088653b14b4"}, {"name": "45e33f15396864", "showlegend": false, "x": [0.0], "y": [95], "type": "scatter", "uid": "3d00ada4-deca-11e8-adc2-0088653b14b4"}, {"name": "9f392b0cf8da3c", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00afd4-deca-11e8-bb75-0088653b14b4"}, {"name": "31628e34b6f4fd", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00b1c8-deca-11e8-abc8-0088653b14b4"}, {"name": "37626caf146ee0", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00b39e-deca-11e8-bdde-0088653b14b4"}, {"name": "89c448a82a6387", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d00b588-deca-11e8-871f-0088653b14b4"}, {"name": "301ca963b049e7", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d00b754-deca-11e8-8e0b-0088653b14b4"}, {"name": "5753f4a44bb751", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00b91e-deca-11e8-b4da-0088653b14b4"}, {"name": "a6e19655c3061d", "showlegend": false, "x": [0.0], "y": [96], "type": "scatter", "uid": "3d00be48-deca-11e8-a454-0088653b14b4"}, {"name": "3de47f270ced23", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00c03a-deca-11e8-8742-0088653b14b4"}, {"name": "272fb0830ccb6d", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d00c410-deca-11e8-8de4-0088653b14b4"}, {"name": "d94e813e9af92b", "showlegend": false, "x": [0.0], "y": [45], "type": "scatter", "uid": "3d00c5be-deca-11e8-ae13-0088653b14b4"}, {"name": "350838a2a4d1f4", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00c6cc-deca-11e8-b807-0088653b14b4"}, {"name": "013c5d688a04c9", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00c7ba-deca-11e8-a991-0088653b14b4"}, {"name": "ad7a9607c49c7c", "showlegend": false, "x": [0.0], "y": [50], "type": "scatter", "uid": "3d00c8b6-deca-11e8-8df9-0088653b14b4"}, {"name": "ba8e3b6eb4dd18", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00c9b0-deca-11e8-b373-0088653b14b4"}, {"name": "cf63cdb22338b7", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00ca9e-deca-11e8-b717-0088653b14b4"}, {"name": "3acb7438aa0d92", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d00cb90-deca-11e8-a781-0088653b14b4"}, {"name": "37f2ef39aff8e5", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d00cc8a-deca-11e8-a295-0088653b14b4"}, {"name": "b4afdffc12e7d9", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00cdde-deca-11e8-8506-0088653b14b4"}, {"name": "c9d49b281d5b46", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00cee2-deca-11e8-96f4-0088653b14b4"}, {"name": "bbc98f27acdbbd", "showlegend": false, "x": [0.0], "y": [98], "type": "scatter", "uid": "3d00d006-deca-11e8-a140-0088653b14b4"}, {"name": "abb1ef1385292b", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00d11c-deca-11e8-9792-0088653b14b4"}, {"name": "768ae050834c2e", "showlegend": false, "x": [0.0], "y": [96], "type": "scatter", "uid": "3d00d218-deca-11e8-9b4d-0088653b14b4"}, {"name": "e5012fd079ea73", "showlegend": false, "x": [0.0], "y": [32], "type": "scatter", "uid": "3d00d338-deca-11e8-af25-0088653b14b4"}, {"name": "8ea51d536c4789", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d00d506-deca-11e8-b4c9-0088653b14b4"}, {"name": "683000137f51cb", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d00d68a-deca-11e8-acfc-0088653b14b4"}, {"name": "795c238c5d1bb0", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00d798-deca-11e8-be62-0088653b14b4"}, {"name": "5ef3144ce105ec", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d00d888-deca-11e8-a3f3-0088653b14b4"}, {"name": "2cf8b1591815a4", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d00d982-deca-11e8-971a-0088653b14b4"}, {"name": "814e5bf15712a3", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d00da74-deca-11e8-b893-0088653b14b4"}, {"name": "0fdb1c48878fe5", "showlegend": false, "x": [0.0], "y": [98], "type": "scatter", "uid": "3d00db62-deca-11e8-af08-0088653b14b4"}, {"name": "20f53f00e8e8b4", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00dc52-deca-11e8-b15e-0088653b14b4"}, {"name": "2e4fa515904bd5", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d00dd4c-deca-11e8-a16d-0088653b14b4"}, {"name": "f885db7f728e00", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d00de78-deca-11e8-98bf-0088653b14b4"}, {"name": "9ff2ebd76bc87d", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d00df90-deca-11e8-ab1f-0088653b14b4"}, {"name": "aa67a081409598", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00e080-deca-11e8-b26c-0088653b14b4"}, {"name": "705d6148a9f23a", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d00e17a-deca-11e8-8a59-0088653b14b4"}, {"name": "2114ec50c73379", "showlegend": false, "x": [0.0], "y": [43], "type": "scatter", "uid": "3d00e268-deca-11e8-b5ea-0088653b14b4"}, {"name": "e56262a9a65784", "showlegend": false, "x": [0.0], "y": [95], "type": "scatter", "uid": "3d00e35a-deca-11e8-955c-0088653b14b4"}, {"name": "f8cff4389b7f6b", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00e454-deca-11e8-ba6f-0088653b14b4"}, {"name": "00db18e0c575a8", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00e546-deca-11e8-a720-0088653b14b4"}, {"name": "ecaf2757033f9a", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00e634-deca-11e8-a01f-0088653b14b4"}, {"name": "3cc6b7159024ac", "showlegend": false, "x": [0.0], "y": [94], "type": "scatter", "uid": "3d00e72e-deca-11e8-9a65-0088653b14b4"}, {"name": "68dce0bb15970e", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00e81e-deca-11e8-8ead-0088653b14b4"}, {"name": "2fe8360d29bb96", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00e90c-deca-11e8-8b72-0088653b14b4"}, {"name": "a6ebc2618cfefa", "showlegend": false, "x": [0.0], "y": [24], "type": "scatter", "uid": "3d00ea3a-deca-11e8-b3e0-0088653b14b4"}, {"name": "15a9b66efb9d41", "showlegend": false, "x": [0.0], "y": [90], "type": "scatter", "uid": "3d00eb8c-deca-11e8-af35-0088653b14b4"}, {"name": "77aba02da20d3c", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00ec9c-deca-11e8-9336-0088653b14b4"}, {"name": "bacf88f2caa904", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00ed98-deca-11e8-9087-0088653b14b4"}, {"name": "af7931e92b0aaa", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d00ee86-deca-11e8-b2e8-0088653b14b4"}, {"name": "f88e2a5b51e0b8", "showlegend": false, "x": [0.0], "y": [31], "type": "scatter", "uid": "3d00ef80-deca-11e8-a3a7-0088653b14b4"}, {"name": "44b85eb9dfdfd2", "showlegend": false, "x": [0.0], "y": [97], "type": "scatter", "uid": "3d00f070-deca-11e8-b97f-0088653b14b4"}, {"name": "802c983725383f", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00f168-deca-11e8-98ad-0088653b14b4"}, {"name": "cb07c10986104a", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d00f25a-deca-11e8-a100-0088653b14b4"}, {"name": "37a03ea0f7883e", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00f354-deca-11e8-a341-0088653b14b4"}, {"name": "1563f1cffeb263", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d00f446-deca-11e8-ae91-0088653b14b4"}, {"name": "c2c40e154c1087", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d00f534-deca-11e8-85fb-0088653b14b4"}, {"name": "41182a11a62457", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00f62e-deca-11e8-a079-0088653b14b4"}, {"name": "80726a34c90120", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d00f71e-deca-11e8-8fb9-0088653b14b4"}, {"name": "491c32049196fb", "showlegend": false, "x": [0.0], "y": [96], "type": "scatter", "uid": "3d00f84a-deca-11e8-a8b1-0088653b14b4"}, {"name": "7456f9b69d0091", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d00f962-deca-11e8-b193-0088653b14b4"}, {"name": "3df2104d34c390", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d00fa52-deca-11e8-b535-0088653b14b4"}, {"name": "56e48676d48f1b", "showlegend": false, "x": [0.0], "y": [47], "type": "scatter", "uid": "3d00fb42-deca-11e8-8657-0088653b14b4"}, {"name": "a908c6634cd3cf", "showlegend": false, "x": [0.0], "y": [96], "type": "scatter", "uid": "3d00fc3a-deca-11e8-b454-0088653b14b4"}, {"name": "1587c850c43fc6", "showlegend": false, "x": [0.0], "y": [45], "type": "scatter", "uid": "3d00fd7a-deca-11e8-8017-0088653b14b4"}, {"name": "c2810d14445200", "showlegend": false, "x": [0.0], "y": [95], "type": "scatter", "uid": "3d00fe80-deca-11e8-8c3c-0088653b14b4"}, {"name": "481b5a6162869d", "showlegend": false, "x": [0.0], "y": [96], "type": "scatter", "uid": "3d00ff70-deca-11e8-878e-0088653b14b4"}, {"name": "90df47dcef697c", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d010068-deca-11e8-a595-0088653b14b4"}, {"name": "395063c6b7c3cb", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d01015a-deca-11e8-868f-0088653b14b4"}, {"name": "219f08d77c7f3e", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d01024a-deca-11e8-9b3a-0088653b14b4"}, {"name": "575f423060a72e", "showlegend": false, "x": [0.0], "y": [26], "type": "scatter", "uid": "3d010346-deca-11e8-b886-0088653b14b4"}, {"name": "e4cedbf4e10220", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d010434-deca-11e8-8227-0088653b14b4"}, {"name": "c3837e121c8c6b", "showlegend": false, "x": [0.0], "y": [49], "type": "scatter", "uid": "3d010562-deca-11e8-8518-0088653b14b4"}, {"name": "c9bbf8d7c04ac2", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d01066e-deca-11e8-9c7e-0088653b14b4"}, {"name": "72a56f63e4aa42", "showlegend": false, "x": [0.0], "y": [30], "type": "scatter", "uid": "3d010768-deca-11e8-945b-0088653b14b4"}, {"name": "8542c17b332d16", "showlegend": false, "x": [0.0], "y": [46], "type": "scatter", "uid": "3d010858-deca-11e8-af78-0088653b14b4"}, {"name": "30a90aba661212", "showlegend": false, "x": [0.0], "y": [29], "type": "scatter", "uid": "3d010952-deca-11e8-81f1-0088653b14b4"}, {"name": "247310ae58d695", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d010a4c-deca-11e8-9488-0088653b14b4"}, {"name": "4d05db6442ac6a", "showlegend": false, "x": [0.0], "y": [31], "type": "scatter", "uid": "3d010b3a-deca-11e8-8352-0088653b14b4"}, {"name": "759b6e9597c2f7", "showlegend": false, "x": [0.0], "y": [27], "type": "scatter", "uid": "3d010c36-deca-11e8-b7c5-0088653b14b4"}, {"name": "aa52205b0755c1", "showlegend": false, "x": [0.0], "y": [48], "type": "scatter", "uid": "3d010d26-deca-11e8-a227-0088653b14b4"}, {"name": "1a3aa702664949", "showlegend": false, "x": [0.0], "y": [28], "type": "scatter", "uid": "3d010e18-deca-11e8-89f5-0088653b14b4"}], {"legend": {"orientation": "h"}, "title": "Killing is Winning?", "xaxis": {"title": "No Damage"}, "yaxis": {"title": "Rank (Lower is Better)"}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>

