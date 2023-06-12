# Projeto GymGo
## Projeto machine learning 7º periodo de engenharia de software - UniBrasil

### Estruturas de pasta do projeto
O projeto é composto de 5 pastas principais

- Estudando algoritmos
- Executar
- Homologacao
- referencias
- videos

##### Estudando Algoritmos
Nesse diretório estão todos os algoritmos que foram analisados e aplicados até encontrar o que esta sendo utilizado como definitivo no projeto, essa pasta se mantem somente para motivos de histórico

##### Executar
Nesse diretório estão os arquivos que devem ser executados para o funcionamento do projeto, no mesmo existem mais dois diretorios, um para a verificação do exercicio Polichinelo e outro para o exercicio Desenvolvimento, ambas as pastas apresentam dois arquivos internos no qual um deles, chamado executarCamera, irá realizar a comparação utilizando sua camera como parametro de captura de video, e o outro arquivo, chamado de executarVideo, é para utilizar um video simulando a captura pela camera

##### Homologacao
Esse diretório armazena todos os registros de features que tentamos adicionar no projeto final sem sucesso, arquivos mantidos somente para registros de histórico 

##### referencias
Esse diretório armazena arquivos de referencia, no qual estão registrados todos os pontos corporais capturados nos videos de aprendizado, essas informações são utilizadas para verificar se o movimento capturado pela camera ou video de comparação estão sendo realizado de maneira acertiva

##### videos
Esse diretório armazena todos os videos utilizados no projeto, tanto os videos de aprendizado quanto os videos de compração

### Como funciona a aplicação

- Certifique de ter todas as dependências e bibliotecas baixadas em seu computador

O objetivo do projeto é verificar se o usuário está realizando os exercicios de maneira correta, para isso ser possível é necessário ter um arquivo de referencia que armazene os valores corretos e seja utilizado para realizar as comparações com o video capturado pela camera ou video de referencia

Ao realizar a comparação será informado na tela um valor que quanto mais próximo de zero mais correta esta a axecução do exercicio em realação ao video de aprendizado

Caso não tenha um arquivo de referencia no path informado no código o memso irá abrir o video de aprendizado e realizar a captura dos valores e armazenar em um arquivo de refencia, com essas informações ele irá seguir com a comparação da camera ou video de compração

> **Para encerrar a captura da imagem ou vídeo de referencia pressione Q do teclado**
