#NOME: JÁDYLA MARIA CESÁRIO FIRMINO
#7º PERÍODO - ENGENHARIA DE COMPUTAÇÃO
#APRENDIZADO DE MÁQUINA - KOBE BRYANT

library("tidyverse")
library("readr")
library("class")
library("base")
library("dplyr")
library("tree")
library("RSNNS")


treinamento <- read_csv("treino.csv")
testeFinal <- read_csv("teste.csv")
separaTreino <- treinamento

#--------------------------------------------------------------------------------------------------------
#### VARIÁVEIS GLOBAIS ####
K <- NULL
result <- NULL

VPknn <- 0
FNknn <- 0
FPknn <- 0
VNknn <- 0

VPknnTeste <- 0
FNknnTeste <- 0
FPknnTeste <- 0
VNknnTeste <- 0

VPbase <- 0
FNbase <- 0
FPbase <- 0
VNbase <- 0

VPbaseTeste <- 0
FNbaseTeste <- 0
FPbaseTeste <- 0
VNbaseTeste <- 0

VPad <- 0
FNad <- 0
FPad <- 0
VNad <- 0

VPadTeste <- 0
FNadTeste <- 0
FPadTeste <- 0
VNadTeste <- 0

VPmlp <- 0
FNmlp <- 0
FPmlp <- 0
VNmlp <- 0

VPmlpTeste <- 0
FNmlpTeste <- 0
FPmlpTeste <- 0
VNmlpTeste <- 0

#=====================================
#aqui foram declaradas as variáveis 
#globais
#=====================================




#--------------------------------------------------------------------------------------------------------
#### MATRIZ DE CONFUSÃO ####
matrizConfusao <- function(targetTest, result, VP, VN, FP, FN, tam){
  for (i in 1:tam){
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
#=====================================
#aqui serão definidos os valores para
#a matriz de confusão, para cada algo-
#ritmo que chamar a função
#=====================================




#--------------------------------------------------------------------------------------------------------
#### MÉTRICAS DE ANÁLISE ####

precisao <- function(VP, FP, nome){
  prec = VP / (VP + FP)
  print(paste("Precisao", nome, "=", prec))
}
revocacao <- function(VP, FN, nome){
  rec = VP / (VP + FN)
  print(paste("Recall", nome, "=", rec))
}




#--------------------------------------------------------------------------------------------------------
#### BASELINE ####
subset (table(testeFinal$shot_made_flag), table(testeFinal$shot_made_flag) == max (table (testeFinal$shot_made_flag)))
vetorBaseline <- testeFinal$shot_made_flag
for (i in seq_along(vetorBaseline)) {
  if (vetorBaseline[i] == 1){
    vetorBaseline[i] = 0
  }
}
print(vetorBaseline)
valoresBase <- matrizConfusao(testeFinal$shot_made_flag, vetorBaseline, VPbase, VNbase, FPbase, FNbase, 2034)
VPbase =+ valoresBase$VP
VNbase =+ valoresBase$VN
FNbase =+ valoresBase$FN
FPbase =+ valoresBase$FP

print(paste("VP=", VPbase))
print(paste("VN=", VNbase))
print(paste("FN=", FNbase))
print(paste("FP=", FPbase))
#=====================================
#aqui os valores foram trocados porque
#o zero é a classe majoritária, então
#foi considerado o VP (calculado como VN)
#=====================================
precisao(VNbase, FNbase, "BASELINE")
revocacao(VNbase, FPbase,"BASELINE")



#--------------------------------------------------------------------------------------------------------
#### KNN ####
melhorK <- function(train, test, targetTrain, targetTest, K){
  print("ERROS:")
  menor <- 1
  cont <- NULL
  erro <- NULL
  
  for (i in 1:50) {
    result <- knn(train = train, test = test, cl = targetTrain, k = i)
    erro[i] <- mean(targetTest != result)
    print(paste("K = " , i , ":" , erro[i]))
    
    if (menor > erro[i]){
      menor = erro[i]
      cont = i
    }
  }
  
  return(cont)
}
#=====================================
#a partir da primeira partição determinar 
#o melhor valor de K dentre os 50 vizinhos 
#mais próximos
#=====================================


knnFunction <- function(train, test, targetTrain, targetTest, K){
  result <- knn(train = train, test = test, cl = targetTrain, k = K)
  ##print(result)
  return(result)
}
#=====================================
#função que implementa o KNN
#=====================================




#--------------------------------------------------------------------------------------------------------
#### ÁRVORE DE DECISÃO ####
treeFunction <- function(train, test, targetTrain){
  model <- tree(train ~ ., targetTrain)
}
#=====================================
#função que implementa a AD
#=====================================




#--------------------------------------------------------------------------------------------------------
#### MLP ####
mlpFunction <- function(train, test, targetTrain, targetTest){
  model <- mlp(	x = train, 
                y = targetTrain, 
                size = 5, 
                learnFuncParams = c(0.1), 
                maxit = 100, 
                inputsTest = test, 
                targetsTest = targetTest)
}
#=====================================
#função que implementa o MLP
#=====================================




#--------------------------------------------------------------------------------------------------------
#### CROSS VALIDATION ####

#=====================================
#'for' que vai separar as partições
#foi ecolhido um k de 7 (7 partições)
#resultando então em 1194 observações
#para teste em cada uma 
#=====================================
for (i in 1:7) {
  partTeste <- separaTreino[sample(nrow(separaTreino), 1194), replace = FALSE]
  part <- setdiff(treinamento, partTeste)
  
  separaTreino <- setdiff(separaTreino, partTeste)
  
  alvo <- part$shot_made_flag
  alvoTeste <- partTeste$shot_made_flag
  part$shot_made_flag <- NULL
  partTeste$shot_made_flag <- NULL
  
  #===================================
  #vai analisar o melhor K a partir da 
  #primeira partição
  #===================================
  if(i == 1){
    K <- melhorK(part, partTeste, alvo, alvoTeste, K)
    print(paste("MELHOR K: " , K))
  }
  
  #==================================
  #'result' vai receber umvetor com a
  #resposta do treinamento
  #==================================
  result <- knnFunction(part, partTeste, alvo, alvoTeste, K)
  
  #==================================
  #definindo os valores da matriz de
  #confusão
  #==================================
  valores <- matrizConfusao(alvoTeste, result, VPknn, VNknn, FPknn, FNknn, 1194)
  VPknn =+ valores$VP
  VNknn =+ valores$VN
  FNknn =+ valores$FN
  FPknn =+ valores$FP
  
  print(paste("KNN - particao", i))
  print(paste("VP=", valores$VP))
  print(paste("VN=", valores$VN))
  print(paste("FN=", valores$FN))
  print(paste("FP=", valores$FP))
  
  #treeFunction(part, partTeste, alvo)
  #mlpFunction(part, partTeste, alvo, alvoTeste)
  print("---------------------------------------------------------------------------")
}

precisao(VPknn, FPknn, "VALIDAÇÂO KNN")
revocacao(VPknn, FNknn, "VALIDAÇÂO KNN")




#--------------------------------------------------------------------------------------------------------
#### TESTE REAL ####
treinamentoSemSaida <- treinamento
treinamentoFinalSaida <- treinamentoSemSaida$shot_made_flag
treinamentoSemSaida$shot_made_flag <- NULL

testeFinalSemSaida <- testeFinal
testeFinalSaida <- testeFinalSemSaida$shot_made_flag
testeFinalSemSaida$shot_made_flag <- NULL

## KNN
resultKnnTeste <- knnFunction(treinamentoSemSaida, testeFinalSemSaida, treinamentoFinalSaida, testeFinalSaida, K)
valoresKnnTesteFinal <- matrizConfusao(testeFinalSaida, resultKnnTeste, VPknnTeste, VNknnTeste, FPknnTeste, FNknnTeste, 2034)
VPknnTeste =+ valoresKnnTesteFinal$VP
VNknnTeste =+ valoresKnnTesteFinal$VN
FNknnTeste =+ valoresKnnTesteFinal$FN
FPknnTeste =+ valoresKnnTesteFinal$FP
print(paste("VP KNN teste final=", VPknnTeste))
print(paste("VN KNN teste final=", VNknnTeste))
print(paste("FN KNN teste final=", FNknnTeste))
print(paste("FP KNN teste final=", FPknnTeste))
precisao(VPknnTeste, FPknnTeste, "TESTE FINAL KNN")
revocacao(VPknnTeste, FNknnTeste, "TESTE FINAL KNN")




#--------------------------------------------------------------------------------------------------------
#TÓPICO 7 - ANÁLISE DOS RESULTADOS ALGORITMO BASELINE




