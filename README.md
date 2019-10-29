# Live D'Wine

O consumo de vinhos tem sido uma tendência de consumo na categoria de alcoólicos desde os últimos anos. De acordo com o IBRAVIN (Instituto Brasileiro de Vinho), o Brasil é 20º país mais consumidor de vinho do mundo, onde o consumo per capita foi estimado em 2 litros.  

Com base nessas estatísticas, é correto afirmar que quanto maior o consumo de vinhos, mais exigente teremos o consumidor, em busca de melhores recomendações.

O Live D’Wine vem para solucionar este problema do consumidor que não possui grandes experiências experiências neste ramo e possui interesse em se aventurar nessse mundo e aprimorar seu paladar.

## Filtros Colaborativos
Esse tipo de filtro é baseado nas taxas dos usuários e nos recomenda vinhos que ainda não experimentamos, mas usuários semelhantes a nós aprovaram. Para determinar se dois usuários são semelhantes ou não, esse filtro considera os vinhos classificados. Observando os itens em comum, esse tipo de algoritmo basicamente prevê a taxa similaridade de um vinho para um usuário que ainda não o experimentou, com base nas taxas de usuários semelhantes.

#### Similaridade Cosseno
Todos conhecemos os vetores: eles podem ser 2D, 3D ou o que for. Vamos pensar em 2D por um momento, porque é mais fácil imaginar em nossa mente, e vamos atualizar primeiro o conceito de produto escalar . O produto escalar entre dois vetores é igual à projeção de um deles no outro. Portanto, o produto escalar entre dois vetores idênticos (ou seja, com componentes idênticos) é igual ao seu módulo quadrado, enquanto que se os dois forem perpendiculares (ou seja, eles não compartilham nenhuma direção), o produto escalar é zero. Geralmente, para vetores n- dimensionais, o produto escalar pode ser calculado como mostrado abaixo.

![equação_cosseno](https://miro.medium.com/max/968/1*xD_jLdpqESuOaBFXyV9YGA.png)

O produto escalar é importante ao definir a semelhança, pois está diretamente conectado a ele. A definição de similaridade entre dois vetores u e v é, de fato, a razão entre o produto escalar e o produto de suas magnitudes.

![fórmula_similaridade_cosseno](https://miro.medium.com/max/776/1*r5ULMbx7ju3_Y4TU1PJIyQ.png)

Aplicando a definição de similaridade, isso será de fato igual a 1 se os dois vetores forem idênticos e será 0 se os dois forem ortogonais. Em outras palavras, a semelhança é um número delimitado entre 0 e 1 que indica quanto os dois vetores são semelhantes.

