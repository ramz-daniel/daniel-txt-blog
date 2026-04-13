+++
date = '2026-02-01T19:57:41-07:00'
draft = true
title = 'Matrices y Forward de una Red'
+++

## ¿Qué es?
Una matriz es un arreglo rectangular de números -reales o complejos- donde llamamos *renglón* a cada fila horizontal de números y *columna* a cada fila vertical; y los enumeramos de arriba a abajo, y de izquierda a derecha, respectivamente. Por ejemplo, en la matriz $A$
$$
A=\begin{pmatrix}
1 & 2 & 3 \\\\
4 & 5 & 6
\end{pmatrix},
$$
los renglones, escritos en orden, son los vectores
$$
\begin{pmatrix}
1 & 2 & 3
\end{pmatrix}\quad 
\begin{pmatrix}
4 & 5 & 6
\end{pmatrix}
$$
y las columnas, también en orden, son
$$
\begin{pmatrix}
1 \\\\
4
\end{pmatrix}
\quad
\begin{pmatrix}
2 \\\\
5
\end{pmatrix}
\quad
\begin{pmatrix}
3 \\\\
6
\end{pmatrix}
$$
## Posición dentro de la matriz.
En la matriz, cada número individual podemos identificarlo con la posición que ocupa. Escribimos $a_{ij}$ para referimos al elemento de la matriz $A$ que se encuentra en la intersección de el renglón $i$ y la columna $j$. Por ejemplo, en la matriz anterior, $a_{21}=4$, porque es el elemento que se encuentra en la intersección entre el segundo renglón y la primera columna
$$
A=\begin{pmatrix}
\textcolor{red}1 & 2 & 3 \\\\
\textcolor{red}4 & \textcolor{red}5 & \textcolor{red}6
\end{pmatrix}.
$$

## Multiplicación de matrices

La multiplicación de matrices se efectúa multplicando *renglón* $\times$ *columna*, es decir, para obtener la entrada $r_{ij}$ del resultado debemos hacer un producto entre el renglón $i$ de la primera matriz y la columna $j$ de la segunda.
Por ejemplo, para multiplicar
$$
B \cdot C = \begin{pmatrix}
2 & 1 \\\\ 3&2
\end{pmatrix}
\cdot
\begin{pmatrix}
2 & 1 \\\\ 4&3
\end{pmatrix}
$$ 
obtendremos el elemento que corresponde a la posición renglón-$1$ columna-$1$ de la matriz resultado multiplicando el renglón $1$ de $B$ por la columna $1$ de $C$

$$
\begin{pmatrix}
2&2
\end{pmatrix}
\cdot
\begin{pmatrix}
2\\\\4
\end{pmatrix}
= 2 \cdot 2 + 1 \cdot 4 =8
$$
$$
\implies B \cdot C =
\begin{pmatrix}
8& \\\\
 &
\end{pmatrix}
$$
obtendremos el elemento que corresponde a la posición renglón-$2$ columna-$1$ multiplicando el renglón $2$ de la matriz $B$ por la columna $1$ de $C$

$$
\begin{pmatrix}
3 & 2
\end{pmatrix}
\cdot
\begin{pmatrix}
2 \\\\
4
\end{pmatrix}
= 3 \cdot 2 + 2 \cdot 4 = 14
$$
$$
\implies B \cdot C = 
\begin{pmatrix}
8 & 14 \\\\ &
\end{pmatrix}
$$
Y así sucesivamente hasta tener el resultado
$$
B \cdot C = 
\begin{pmatrix}
8 & 14 \\\\ 5 & 9
\end{pmatrix}
$$

## ¿Porqué se usa en una Red Neuronal?
La forma más sencilla de una red neuronal se llama *Perceptrón*. Este consiste de una serie de entradas que se conectan a una única neurona, y donde cada una de las aristas que conectan las entradas con la neurona posee un *Peso*.
La razón para usar la multiplicación de matrices es que la forma en que los pesos se combinan con las entradas para producir un resultado en la neurona es similar a la combinación renglón-columna de la multiplicación de matrices.  La $Salida$ del perceptrón se calcula multiplicando cada entrada por la respectiva arista que conecta con la neurona y, al final, sumar los resultados
$$
\alpha = x\cdot a+ y\cdot b+ z\cdot c
$$
Esto puede representarse en forma matricial como
$$
\alpha = \begin{pmatrix}
x & y & z
\end{pmatrix} \cdot
\begin{pmatrix}
a \\\\
b \\\\
c
\end{pmatrix}
$$
Para una red neuronal levemente más compleja, con $2$ neuronas, el procedimiento se ve así
$$
\begin{cases}
\alpha = x\cdot a + y\cdot b + z \cdot c \\\\
\beta = x\cdot d + y\cdot e + z\cdot f
\end{cases}
$$
que en notación matricial corresponde a la multiplicación
$$
\begin{pmatrix}
\alpha & \beta
\end{pmatrix}
=\begin{pmatrix}
x & y & z
\end{pmatrix}
\cdot
\begin{pmatrix}
a & d \\\\
b & e \\\\
c & f
\end{pmatrix}
$$

## ¿Cómo se implementa el Feed Forward de una Red Neuronal?
El Feed Forward de una red neuronal es una repetición secuencial del principio anterior: Si tenemos una red con $n$ cantidad de capas entonces empezaremos con las entradas, efectuando la multimplicación matricial, y al obtener el resultado este pasará a ser la entrada para la siguiente capa de la red, y se repetirá este proceso hasta terminar todas las capas y obtener la salida de la red completa. 

El siguiente código en Python corresponde a la clase de una única capa que ejecuta la transformación matricial de la que hablamos antes:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__() 
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) 

    def forward(self, x):
        return F.linear(x, self.weight) 

```
Veamos qué hace cada parte de este código. Primero,
 ```python
def __init__(self, in_features, out_features):
    super().__init__() 
```
inicializa la clase, heredandole atributos de la clase $nn.Module$. Después se asignan aleatoriamente los pesos de las aristas:
```python
self.weight = nn.Parameter(torch.randn(out_features, in_features)) 
```
Y finalmente, se llama al método $F.linear$ de Pytorch para calcular las salidas de la capa:
```python
def forward(self, x):
    return F.linear(x, self.weight) 
```
Este método realiza *transformaciones afines*, que son precisamente las de forma $y=xA^T + b$.