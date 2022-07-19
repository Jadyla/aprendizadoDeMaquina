#NOME: JÁDYLA MARIA CESÁRIO FIRMINO
#7º PERÍODO - ENGENHARIA DE COMPUTAÇÃO
#APRENDIZADO DE MÁQUINA - KOBE BRYANT

#install.packages("class")
library("tidyverse")
library("readr")
library("class")
library("base")
library("dplyr")


treinamento <- read_csv("treino.csv")
separaTreino <- treinamento

#--------------------------------------------------------------------------
#TÓPICO 1 - CROSS-VALIDATION
#serão separadas 42 partições com 199 instâncias cada uma, sendo 159 para treinamento e 40 para teste, e esta
#separação será feito por um 'for' que percorrerá as linhas

for (i in 1:2) {
  part <- separaTreino[sample(nrow(separaTreino), 159), replace = FALSE]
  separaTreino <- setdiff(separaTreino, part)
  partTeste <- separaTreino[sample(nrow(separaTreino), 40), replace = FALSE]
  separaTreino <- setdiff(separaTreino, partTeste)
  View(separaTreino)
}
  



#--------------------------------------------------------------------------
#TÓPICO 2 - MÉTRICAS DE ANÁLISE



#--------------------------------------------------------------------------
#TÓPICO 3 - DEFINIÇÃO DO ALGORITMO BASELINE



#--------------------------------------------------------------------------
#TÓPICO 4 - KNN



#--------------------------------------------------------------------------
#TÓPICO 5 - DECISION TREE



#--------------------------------------------------------------------------
#TÓPICO 6 - MLP (Multilayer Perceptron)



#--------------------------------------------------------------------------
#TÓPICO 7 - ANÁLISE DOS RESULTADOS ALGORITMO BASELINE



#--------------------------------------------------------------------------
#TÓPICO 3 - ANÁLISE KNN, DECISION TREE E MLP
