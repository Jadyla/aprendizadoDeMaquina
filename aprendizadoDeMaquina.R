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
library("rpart")


treinamento <- read_csv("treino.csv")
testeFinal <- read_csv("teste.csv")
separaTreino <- treinamento

#--------------------------------------------------------------------------------------------------------
#### VARIÁVEIS GLOBAIS ####
K <- NULL
L <- NULL
result <- NULL
resultMlp <- NULL
resultAd <- NULL

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
##print(vetorBaseline)
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
    ##print(paste("K = " , i , ":" , erro[i]))
    
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
  predVal <- NULL
  fit <- rpart(targetTrain ~ ., data = train, method="class")
  #summary(fit)
  predsVal <- predict(fit, test)
  
  predVal <- (predsVal[,1])
  auxVal <- ifelse (predVal > 0.50, 0, 1)
  
  return(as.numeric(auxVal))
}
#=====================================
#função que implementa a AD
#=====================================




#--------------------------------------------------------------------------------------------------------
#### MLP ####
mlpFunction <- function(train, test, targetTrain, targetTest){
  model <- mlp(	x = train, 
                y = targetTrain, 
                size = 10, 
                learnFuncParams = c(0.03), 
                maxit = 50, 
                inputsTest = test, 
                targetsTest = targetTest)
  predsVal <- predict(model,test)
  predVal <- ifelse (predsVal > 0.508, 1, 0)
  return(predVal)
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
  #'result' vai receber um vetor com a
  #resposta do treinamento
  #==================================
  result <- knnFunction(part, partTeste, alvo, alvoTeste, K)
  
  #==================================
  #definindo os valores da matriz de
  #confusão KNN
  #==================================
  valores <- matrizConfusao(alvoTeste, result, VPknn, VNknn, FPknn, FNknn, 1194)
  VPknn =+ valores$VP
  VNknn =+ valores$VN
  FNknn =+ valores$FN
  FPknn =+ valores$FP
  #print(as.matrix(part))
  
  part$combined_shot_type <- as.numeric(part$combined_shot_type)
  part$shot_zone_area <- as.numeric(part$shot_zone_area)
  part$shot_zone_basic <- as.numeric(part$shot_zone_basic)
  part$opponent <- as.numeric(part$opponent)
  
  partTeste$combined_shot_type <- as.numeric(partTeste$combined_shot_type)
  partTeste$shot_zone_area <- as.numeric(partTeste$shot_zone_area)
  partTeste$shot_zone_basic <- as.numeric(partTeste$shot_zone_basic)
  partTeste$opponent <- as.numeric(partTeste$opponent)
  #print(as.matrix(partTeste))
  
  resultMlp <- mlpFunction(as.matrix(part), partTeste, alvo, alvoTeste)
  valoresMlp <- matrizConfusao(alvoTeste, resultMlp, VPmlp, VNmlp, FPmlp, FNmlp, 1194)
  VPmlp =+ valoresMlp$VP
  VNmlp =+ valoresMlp$VN
  FNmlp =+ valoresMlp$FN
  FPmlp =+ valoresMlp$FP
  #print(alvoTeste)
  
  resultAd <- treeFunction(part, partTeste, alvo)
  valoresAd <- matrizConfusao(alvoTeste, resultAd, VPad, VNad, FPad, FNad, 1194)
  VPad =+ valoresAd$VN
  VNad =+ valoresAd$VP
  FNad =+ valoresAd$FP
  FPad =+ valoresAd$FN
  #print(alvoTeste)-------------
  
  print(paste("particao", i))
  print("KNN")
  print(paste("VP=", VPknn))
  print(paste("VN=", VNknn))
  print(paste("FN=", FNknn))
  print(paste("FP=", FPknn))
  print("-------------------------------------")
  print("AD")
  print(paste("VP=", VPad))
  print(paste("VN=", VNad))
  print(paste("FN=", FNad))
  print(paste("FP=", FPad))
  print("-------------------------------------")
  print("MLP")
  print(paste("VP=", VPmlp))
  print(paste("VN=", VNmlp))
  print(paste("FN=", FNmlp))
  print(paste("FP=", FPmlp))
  
  #treeFunction(part, partTeste, alvo)
  #mlpFunction(part, partTeste, alvo, alvoTeste)
  print("################################################################")
}

precisao(VPknn, FPknn, "VALIDAÇÂO KNN")
revocacao(VPknn, FNknn, "VALIDAÇÂO KNN")

precisao(VPmlp, FPmlp, "VALIDAÇÂO MLP")
revocacao(VPmlp, FNmlp, "VALIDAÇÂO MLP")

precisao(VPad, FPad, "VALIDAÇÂO AD")
revocacao(VPad, FNad, "VALIDAÇÂO AD")


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


## MLP
treinamentoSemSaida$combined_shot_type <- as.numeric(treinamentoSemSaida$combined_shot_type)
treinamentoSemSaida$shot_zone_area <- as.numeric(treinamentoSemSaida$shot_zone_area)
treinamentoSemSaida$shot_zone_basic <- as.numeric(treinamentoSemSaida$shot_zone_basic)
treinamentoSemSaida$opponent <- as.numeric(treinamentoSemSaida$opponent)

testeFinalSemSaida$combined_shot_type <- as.numeric(testeFinalSemSaida$combined_shot_type)
testeFinalSemSaida$shot_zone_area <- as.numeric(testeFinalSemSaida$shot_zone_area)
testeFinalSemSaida$shot_zone_basic <- as.numeric(testeFinalSemSaida$shot_zone_basic)
testeFinalSemSaida$opponent <- as.numeric(testeFinalSemSaida$opponent)


resultMlpTeste <- mlpFunction(treinamentoSemSaida, testeFinalSemSaida, treinamentoFinalSaida, testeFinalSaida)
valoresMlpTesteFinal <- matrizConfusao(testeFinalSaida, resultMlpTeste, VPmlpTeste, VNmlpTeste, FPmlpTeste, FNmlpTeste, 2034)
VPmlpTeste =+ valoresMlpTesteFinal$VP
VNmlpTeste =+ valoresMlpTesteFinal$VN
FNmlpTeste =+ valoresMlpTesteFinal$FN
FPmlpTeste =+ valoresMlpTesteFinal$FP
print(paste("VP MLP teste final=", VPmlpTeste))
print(paste("VN MLP teste final=", VNmlpTeste))
print(paste("FN MLP teste final=", FNmlpTeste))
print(paste("FP MLP teste final=", FPmlpTeste))
precisao(VPmlpTeste, FPmlpTeste, "TESTE FINAL MLP")
revocacao(VPmlpTeste, FNmlpTeste, "TESTE FINAL MLP")


## AD
resultAdTeste <- treeFunction(treinamentoSemSaida, testeFinalSemSaida, treinamentoFinalSaida)
valoresAdTesteFinal <- matrizConfusao(testeFinalSaida, resultAdTeste, VPadTeste, VNadTeste, FPadTeste, FNadTeste, 2034)
VPadTeste =+ valoresAdTesteFinal$VP
VNadTeste =+ valoresAdTesteFinal$VN
FNadTeste =+ valoresAdTesteFinal$FN
FPadTeste =+ valoresAdTesteFinal$FP
print(paste("VP AD teste final=", VPadTeste))
print(paste("VN AD teste final=", VNadTeste))
print(paste("FN AD teste final=", FNadTeste))
print(paste("FP AD teste final=", FPadTeste))
precisao(VPadTeste, FPadTeste, "TESTE FINAL AD")
revocacao(VPadTeste, FNadTeste, "TESTE FINAL AD")