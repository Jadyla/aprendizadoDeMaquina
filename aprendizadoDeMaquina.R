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

#declaração das variáveis a serem utilizadas (declaradas aqui porque precisam ser globais)
K <- NULL
VP <- 0
FN <- 0
FP <- 0
VN <- 0
result <- NULL

#a partir da primeira partição determinar o melhor valor de K dentre os 50 vizinhos mais próximos
melhorK <- function(train, test, targetTrain, targetTest, K){
  print("ERROS:")
  menor <- 1
  cont <- NULL
  erro <- NULL
  #vai analisar o percentual de erro para cada valor de k, variando de 1 a 30
  for (i in 1:50) {
    result <- knn(train = train, test = test, cl = targetTrain, k = i)
    erro[i] <- mean(targetTest != result)
    print(paste("K = " , i , ":" , erro[i]))
    #compara se o valor do erro é menor, apenas para retornar o valor do melhor k
    if (menor > erro[i]){
      menor = erro[i]
      cont = i
    }
  }
  #retorna o melhor k para ser usado para todas as partições
  return(cont)
}

#funcao para determinar os valores para matriz de confusão
#aqui vai ser comparado os valores resultados do treinamento e os valores esperados
#após análise os valores serão somados e retornarão para a função mãe para serem incrementados
matrizConfusao <- function(targetTest, result, VP, VN, FP, FN){
  for (i in 1:1194){
    if (targetTest[i] == 1 && result[i] == 1){
      VP = VP + 1
    }
    if (targetTest[i] == 0 && result[i] == 0){
      VN = VN + 1
    }
    if (targetTest[i] == 1 && result[i] == 0){
      FN = FN + 1
    }
    if (targetTest[i] == 0 && result[i] == 1){
      FP = FP + 1
    }
  }
  return(list(VP = VP, VN = VN, FN = FN, FP = FP))
}


#função que implementa o KNN
knnFunction <- function(train, test, targetTrain, targetTest, K){
  result <- knn(train = train, test = test, cl = targetTrain, k = K)
  ##print(result)
  return(result)
}

#'for' que vai separar as partições
#foi ecolhido um k de 7 (7 partições) resultando então en 1194 observações para teste em cada uma
#método cross validation
for (i in 1:7) {
  partTeste <- separaTreino[sample(nrow(separaTreino), 1194), replace = FALSE]
  part <- setdiff(treinamento, partTeste)
  
  separaTreino <- setdiff(separaTreino, partTeste)
  
  alvo <- part$shot_made_flag
  alvoTeste <- partTeste$shot_made_flag
  part$shot_made_flag <- NULL
  partTeste$shot_made_flag <- NULL
  
  #vai analisar o melhor K a partir da primeira partição
  if(i == 1){
    K <- melhorK(part, partTeste, alvo, alvoTeste, K)
    print(paste("MELHOR K: " , K))
  }
  
  ##print(alvoTeste)
  
  print(paste("K= " , K))
  result <- knnFunction(part, partTeste, alvo, alvoTeste, K)
  
  valores <- matrizConfusao(alvoTeste, result, VP, VN, FP, FN)
  
  VP =+ valores$VP
  VN =+ valores$VN
  FN =+ valores$FN
  FP =+ valores$FP
  
  print(paste("VP=", valores$VP))
  print(paste("VN=", valores$VN))
  print(paste("FN=", valores$FN))
  print(paste("FP=", valores$FP))
  print("-------------------------------------------------")
}

precisao <- function(){
  prec = VP / (VP + FP)
  print(paste("Precisao=", prec))
}
revocacao <- function(){
  rec = VP / (VP + FN)
  print(paste("Recall=", rec))
}

precisao()
revocacao()

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
