![](https://cdn.discordapp.com/attachments/899049208324161548/1202010731734630451/Blood_Circulation.gif?ex=65cbe6ba&is=65b971ba&hm=334523b12e225840e883df458d54fdd2740c5ced25107221b029d8b0fd889f43&)
# Uma visão geral de doença cardiovascular

Este estudo é para mostrar as pessoas que mais tendem a sofrer com ataque cardíaco, determinar os principais alvos e os grupos de riscos, tais como idade, gênero, ambiente, etc... A intenção além de treinar meu conhecimento com análise de dados também é conscientizar e melhorar a prevenção da doença.

### Abreviações e Significados

* **Age** : Idade do Paciente
* **Sex** : Genero do Paciente
* **cp** : Tipo de dor no peito tipo de valor 1: Típico Angina, Valor 2: Angina Atípico Valor 3: Dor não anginosa Valor 4: Assitomática
* **exang**: Exercício que induziu Angina (1 = sim; 0 = não)
* **caa**: Número de vasos sanguineos (0-4)
* **trtbps** : Pressão sanguinea em repouso
* **chol** : Colesterol em miligramas
* **fbs** : (quantidade de açúcar no sangue > 120 mg/dl) (1 = verdade; 0 = falso)
* **rest_ecg** : Eletrocardiograma em repouso    
* **thalach** : Frequência cardiaca máxima


### Visão geral das doenças cardiovasculares

*Não é uma única doença, mas um conjunto de doenças e lesões que afetam o sistema cardiovascular (o coração e os vasos sanguíneos). Existem diversos fatores que causam a doença cardiovasculares, tais como gordura no sangue, açúcar no sangue (diabetes), estresse emocional, pressão alta, idade avançada, hipertensão, diabetes, hipercolesterolemia, tabagismo, histórico familiar e sedentarismo.*

*Você verá nesta análise todo um estudo analítico que aborda esses sintomas, tais como também os sintomas falsos que são gerados por ansiedade e forjam uma falsa sensação de infarto, dormencia no braço esquerdo, dor nas costas, aperto nos peitos, entre outros sintomas parecidos.*

*Angina
A dor associada à DC muito avançada é conhecida como angina e geralmente se apresenta como uma sensação de pressão no peito, dor no braço, dor na mandíbula e outras formas de desconforto.*


#### Bibliotecas usadas importadas abaixo:

```ruby
import pandas as pd
import seaborn as sns
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/heart-attack-analysis-prediction-dataset/heart.csv')
df.head()
```

#### Leitura da data base:

```ruby
df.tail()
```

```ruby
#conferindo a base de dados
df.info()
```

```ruby
plt.figure(figsize=(15,7))
sns.heatmap(df.corr(), annot=True)
```

```ruby
df.isnull().sum()
```


```ruby
df['exng'].value_counts()
```


```ruby
df.shape
```


```ruby
df['sex'].value_counts()
```


```ruby
#Todas as variáveis

sns.set(style="whitegrid")
colors=sns.color_palette("husl", n_colors=len(df.columns))
for i,col in enumerate(df.columns.values):
    plt.subplot(6,4,i+1)
    plt.scatter([i for i in range(303)],df[col].values.tolist())
    plt.title(col)
    fig,ax=plt.gcf(),plt.gca()
    fig.set_size_inches(10,10)
    plt.tight_layout()
plt.show()
```


```ruby
#Definindo maior número de alvos da doença.
sns.countplot(x=df['sex'])
plt.xticks([0,1],['Feminino', 'Masculino'])
plt.show()
```


```ruby
#Listagem da variavel "Idade"
df['age'].value_counts()
```

###Quem são os mais afetados com problemas cardiaco, mais velhos, quem faz ou não exercícios?
###Com qual dos dois gêneros temos mais problemas cardíacos, mulheres ou homens?
###Como podemos evitar problemas cardiacos com o grupo de pessoa mais afetado?

```ruby
df.columns
```


```ruby
print(df.shape)
df.head(10).style.set_properties(**{'background-color': '#FCD805',
                           'color': 'black',
                           'border-color': 'grey'})
```

```ruby
columns_interface = ['age', 'cp', 'chol']
```


```ruby
def dist_box(df, feature=None, size=(25, 5)):
    fig, (ax_hist, ax_box) = plt.subplots(nrows=2, figsize=size, gridspec_kw={"height_ratios": (.8, .2)})

    sns.boxplot(df[feature], orient='h', color='b', ax=ax_box)
    ax_box.set_xticks([])
    ax_box.set_yticks([])
    sns.histplot(df[feature], bins=30, color='b', ax=ax_hist)
    ax_hist.set_xlabel('')
    plt.suptitle(feature, y=1.02, fontsize=16)
    plt.tight_layout()
```


```ruby
df[columns_interface].describe()
```


```ruby
#Visualização global dos casos

for features in columns_interface:
    ax = df.plot.hist(figsize=(7, 4))
```


```ruby
#Visualização em gráfico de idades e dores no peito
#Tendo em vista que 0, 1, 2, 3 são variáveis de 'CP' (chest pain)

fig, axes = plt.subplots(1, 2, figsize=(17, 6), dpi=200)
sns.countplot(y=df['cp'], ax=axes[0])
axes[0].set_title('Dor no peito (cp)')
sns.countplot(x=df['age'], ax=axes[1])
axes[1].set_title('Idade')
plt.tight_layout()
plt.show()
```

```ruby
#Alvos em comparação de gêneros

g = sns.FacetGrid(df, hue="sex",aspect=2)
g.map(sns.kdeplot, 'trtbps', fill=True)
plt.legend(labels=['Masculino', 'Feminino'])
plt.show()
```

```ruby
df.columns
```

```ruby
cate_val=[]
cont_val=[]

for column in df.columns:
    if df[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)
```


```ruby
cate_val
```


```ruby
cont_val
```


```ruby
df.hist(cont_val,figsize=(13,9))
plt.show()
```


```ruby
#Tipos de dores no peito 'CP' legendado

sns.countplot(x=df['cp'])
plt.xticks([0,1,2,3],['angina típica', 'angina atípica', 'dor no peito atípica', 'assintomática'])
plt.xticks(rotation=0)
plt.show()
```


```ruby
df['chol']=df['chol'].fillna(df['chol'].mean())
df['fbs']=df['fbs'].fillna(df['fbs'].mean())
```


```ruby
(df.isnull().sum())*100/len(df)
```


```ruby
plt.figure(figsize=(30,20))
for i in enumerate(df_cat.columns):
    plt.subplot(3, 5, i[0]+1)
    sns.countplot(x=i[1], hue='fbs', data=df_cat)
```


