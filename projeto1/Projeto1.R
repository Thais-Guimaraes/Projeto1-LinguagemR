# FCD - Data Science Academy - Big Data Analytics com R e Microsoft Azure Machine Learning

# Projeto com feedback 1 - Machine Learning em Logística Prevendo o 
# Consumo de Energia de Carros Elétricos


# Definição do problema de negócio:

# A empresa deseja migrar sua frota para carros
# elétricos com o objetivo de reduzir custos.
# antes de tomar a decisão, a empresa gostaria de prever o consumo de energia de
# carros elétricos com base diversos fatores de utilização e características dos
# veículos


# Origem dos dados:  https://data.mendeley.com/datasets/tb9yrptydn/2


# Após o carregamento dos dados, importação das bibliotecas necessárias,
# temos a visualação dos dados, e em seguida , alterações na nomenclatura
# das colunas, porque é recomendado que não haja espaçamento nos nomes das colunas
# Temos a aplicação de remoção de valores ausentes, análise exploratória, 
# feature selection e aplicação de Machine Learning, utilizando métodos ensemble

# Foi identificado que as variáveis possuem multicolinearidade, 
# são assimétricas, e como o conjunto de dados é pequeno optei por 
# não retirar os outliers a princípio, e aplicar random forest e gradiente boosting,que
# não exigem dados simétricos, a árvore de decisão é robusta
# em dados com outliers. 
# em um dos modelos retirei alguns outliers  para verificar a performance


# Definindo diretório de trabalho

setwd("C:/FCD/BigDataRAzure/Cap20_Projetos/projeto1")

# Carregando pacotes

library(readxl)
library(stats)
library(dplyr)
library(psych)
library(Amelia)
library(caTools)
library(randomForest)
library(pastecs)
library(moments)
library(tidyr)
library(ggplot2)
library(corrplot)
library(Hmisc)
library(GGally)
library(gbm)

# Carregando o dataset

dados <- read_excel("dados/FEV-data-Excel.xlsx")

View(dados)
class(dados)
dim(dados) 

# Temos 53 observações e 25 variáveis


# Alterando os nomes das colunas

colunas <- colnames(dados)

colunas[1] <- "Car_name"
colunas[2] <- "Make"
colunas[3] <- "Model"
colunas[4] <- "Min_price"
colunas[5] <- "Engine_power"
colunas[6] <- "Max_torque"
colunas[7] <- "Brakes"
colunas[8] <- "Drive_type"
colunas[9] <- "Battery_capacity"
colunas[10] <- "Range"
colunas[11] <- "Wheelbase"
colunas[12] <- "Length"
colunas[13] <- "Width"
colunas[14] <- "Height"
colunas[15] <- "Min_empty_wt"
colunas[16] <- "Permissable_gross_wt"
colunas[17] <- "Max_load_capacity"
colunas[18] <- "Num_seats"
colunas[19] <- "Num_doors"
colunas[20] <- "Tire_size"
colunas[21] <- "Max_speed"
colunas[22] <- "Boot_capacity"
colunas[23] <- "Acceleration"
colunas[24] <- "Max_dc_charge"
colunas[25] <- "Mean_Energy"

colnames(dados) <- colunas
View(dados)


# Ao olhar para os dados, vi que há dados ausentes, por isso, decidi 
# excluí-los agora, temos 2% de dados missing, como é visto no missmap


# Temos 42 observações completas 
complete_cases <- sum(complete.cases(dados))
complete_cases

missmap(dados)

# a função lapply é da família apply,
# ela aplica uma função em cada linha, retornando o
# resultado em lista
# aqui podemos ver onde estão os valores nulos

lapply(dados, function(dados){
  sum(is.na(dados))
})

# aqui estamos atribuindo o resultado da função "na.omit"
# a uma variável, criando assim, outro dataframe
# a função na.omit irá excluir as linhas em que temos
# dados missing

df <- na.omit(dados)


# De antemão, sabemos que as variáveis Car_name, Make e Model
# são categóricas

# Com a função "unique" vamos identificar outras possíveis
# variáveis categóricas


lapply(dados, function(v){
  unique(v)
  
})


# Variáveis Categóricas:

# Car_name
# Make
# Model
# Brakes
# Drive_type
# Num_seats
# Num_doors
# Tire_size


# variável Alvo está representada em kWh/100 km



# Separar os dados numéricos e categóricos


dados_categoricos <- df[, c(1,2,3,7,8,18,19,20)]
dados_numericos <- df[, -c(1,2,3,7,8,18,19,20)]

View(dados_categoricos)
View(dados_numericos)



# Alterando essas variáveis categóricas para fatores

# mutate é uma funcao da biblioteca dplyr para modificar colunas
# o simbolo %>% "une" os códigos
# across aplicara a funcao

?mutate

df <- df %>%
  mutate(across(c(1,2,3,7,8,18,19,20), factor))
   
View(df)
  

# Verificando se alterou

df %>%
  select(c(1,2,3,7,8,18,19,20)) %>%
  str()
  

# Tabela de contingência das variáveis categóricas

# E verificando a relação das variaveis categoricas
# com a variável target

# basicamente, temos um carro de cada nome 

table(df$Car_name)

# está distribuido por marcas
table(df$Make)
plot(df$Make, df$Mean_Energy)
?plot

# a média de gasto energético para o tipo 1 é maior
# Porém, temos muito mais dados para o tipo 1 do que para o 2

table(df$Brakes)
boxplot(Mean_Energy ~ Brakes, data = df)

# Variável Drive Type

# o tipo 1 2wd tem mais e os tipos 2 e 3 estao quase empatados
table(df$Drive_type)

# o tipo 4wd tem um maior gasto energético
boxplot(Mean_Energy ~ Drive_type, data = df)

# temos poucas representacões para num_seats 2 e 8
# para as categorias 4 e 5 a mediana não foi alterada
# portanto aparentemente essa variável não tem muita influência na média de gastos
# energético


table(df$Num_seats)
boxplot(Mean_Energy ~ Num_seats, data = df)

# as medianas variam com a variável num_doors
table(df$Num_doors)
boxplot(Mean_Energy ~ Num_doors, data = df)


# aparentemente quanto maior o numero do pneu maior
# o gasto energético
table(df$Tire_size)
boxplot(Mean_Energy ~ Tire_size, data = df)



# Verificando variáveis numéricas
View(dados_numericos)

# usando pastecs
estatisticas <- stat.desc(df[, -c(1,2,3,7,8,18,19,20)])

View(estatisticas)

# verificando boxplot das variaveis
# mostrando em grupos, pois temos muitas

# para estas variáveis, temos alguns outliers
# os dados são positivamente assimétricos
# porque a mediana está próxima do primeiro quartil

par(mfrow=c(1,3))
boxplot(df$Min_price, main= "Min Price")
boxplot(df$Engine_power, main= "Engine Power")
boxplot(df$Max_torque, main = "Max Torque")


# a primeira e ultima variavel, aparentemente tbm possuem
# assimetria positiva, a variavel Range parece ser bem distribuida

par(mfrow=c(1,3))
boxplot(df$Battery_capacity, main = "Battery Capacity")
boxplot(df$Range, main="Range WLTP")
boxplot(df$Wheelbase, main= "Wheelbase")


# aparentemente assimetria positiva
par(mfrow=c(1,3))
boxplot(df$Min_empty_wt, main= "Min Empty Weight")
boxplot(df$Length, main = "Length")
boxplot(df$Width, main = "Width")

# height parece ser quase simetrico
# e as demais positivas
par(mfrow=c(1,3))
boxplot(df$Height, main="Height")
boxplot(df$Permissable_gross_wt, main="Permissable gross wt")
boxplot(df$Max_load_capacity, main="Max Load")


# wheelbase parece ter assimetria negativa
# as outras positiva
par(mfrow=c(1,3))
boxplot(df$Num_seats, main= "Wheelbase")
boxplot(df$Max_speed, main="Max Speed")
boxplot(df$Boot_capacity, main = "Boot Capacity")

# aparentemente acelaration assimetria positiva
# max dc charge aparentemente negativa
par(mfrow=c(1,2))
boxplot(df$Acceleration, main="Acceleration")
boxplot(df$Max_dc_charge, main="Max dc charge")

# Verificando o boxplot da variável alvo, a maioria dos dados se concentram 
# abaixo da média (assimetria positiva) e não temos outliers
boxplot(df$Mean_Energy, main= "Mean Energy")



# para confirmar
# Verificando os coeficientes de Assimetria e Curtose


skewness_value <- lapply(dados_numericos, function(v){
  skewness(v)
  
})


kurtosis_value <- lapply(dados_numericos, function(v){
  kurtosis(v)
  
})


skw_kurt <- cbind(skewness_value,kurtosis_value)

# tabela com o skewness e kurtosis
View(skw_kurt)

# skew = 0, a distribuição é simétrica;
# skew > 0, a distribuição é assimétrica positiva (à direita);
# skew < 0, a distribuição é assimétrica negativa (à esquerda).

# Criando lista vazia
lista <- list()

for(i in skw_kurt[,1]){
  if (i == 0){
    lista <- append(lista,'Normal')
  }
  else if  (i > 0){
    lista <- append(lista,'Assimétrica positiva')
  } 
  
  else{
    lista <- append(lista,'Assimétrica negativa')
  }
  
}

skw_kurt <- as.data.frame(skw_kurt)


skw_kurt$Tipo_Assimetria <- lista



# kurtosis = 0 -> curva normal
# Kurtosis > 0 -> baixo achatamento
# kurtosis < 0 -> alto achatamento

lista2 <- list()

for(i in skw_kurt[,2]){
  if (i == 0){
    lista2 <- append(lista2,'Normal')
  }
  else if  (i > 0){
    lista2 <- append(lista2,'Baixo achatamento')
  } 
  
  else{
    lista2 <- append(lista2,'Alto achatamento')
  }
  
}

skw_kurt$TipoKurtosis <- lista2



# os dados com assimetria positiva estão concentrados abaixo da média
# os dados com assimetria negativa estão concentrados acima da média


View(skw_kurt)


# Visualizando graficamente as assimetrias

# pivot dos dados para poder passar o x no ggplot
data_long <- dados_numericos %>%                         
  pivot_longer(colnames(dados_numericos)) %>% 
  as.data.frame()
head(data_long) 

View(data_long)
View(dados_numericos)


ggp2 <- ggplot(data_long, aes(x = value)) +    
  geom_density() + 
  facet_wrap(~ name, scales = "free")
ggp2


# Verificar a Correlação

cor.plot(dados_numericos)

res_correlacao <- cor(dados_numericos)
round(res_correlacao, 2)
 

# gerando a matriz de correlaçao
res_correlacao_matriz <- rcorr(as.matrix(dados_numericos))
res_correlacao_matriz

 ?corrplot
# se o p-valor for maior que 0.05 (sig.level), será deixado em branco (insig)
# o plot será ordenado por agrupamentos, é possível ver
# que os mais correlacionados estão mais próximos
# p.mat é a matriz de p-valor

corrplot(res_correlacao_matriz$r, type="full", order="hclust", 
         p.mat = res_correlacao_matriz$P, sig.level = 0.05, insig = "blank")



# As variáveis "Height" e "Range", tem baixa correlação com a variável alvo
# A maioria das variáveis tem correlação entre si, sendo que estas listadas
# abaixo possuem uma maior correlação entre si

# Max_dc_charge
# Max_torque
# Max_speed
# Min_price
# Engine_power



# vamos aplicar uma transformação nos dados (normalização)
# apesar que nos modelo baseados em árvore de decisão são mais robustos para esta situação


# Aplicar normalização nos dados 

minmax <- function(x){
  return((x - min(x))/(max(x)-min(x)))
}



df2 <- df %>%
  mutate(across(-c(1,2,3,7,8,18,19,20,25), minmax))
View(df2)



ggpairs(dados_numericos[1:5])
ggpairs(dados_numericos[5:10])
ggpairs(dados_numericos[10:17])



# Feature selection

# Utilizando o Random Forest para extrair a importância das
# variáveis, aqui temos o dataset com todas as variáveis (categóricas e numéricas)

modelo_feature_selection_completo <- randomForest(Mean_Energy ~ . , 
                                         data = df2, 
                                         ntree = 100, 
                                         nodesize = 10,
                                         importance = TRUE) # ao criar o modelo, identificar as variaveis mais relevantes

# Importancia das variaveis (completo)
varImpPlot(modelo_feature_selection_completo)


# Aqui temos com valores numéricos
# Quando temos todas as variaveis a variavel boot_capacity 
# aumenta sua importancia
# vemos que as seguintes variaveis aparecem bem colocadas nos dois
# modelos:
# Wheelbase
# Permissable_gross_wt
# Lenght
# Max_load
# Engine_Power

modelo_feature_selection_numerico <- randomForest(Mean_Energy ~ . -Car_name -Make -Model -Brakes -Drive_type -Num_seats -Num_doors -Tire_size, 
                                                  data = df2, 
                                                  ntree = 100, 
                                                  nodesize = 10,
                                                  importance = TRUE) # ao criar o modelo, identificar as variaveis mais relevantes

# Importancia das variaveis numéricas
varImpPlot(modelo_feature_selection_numerico)



# dividindo os dados em treino e teste


dados_split <- sample.split(df2$Mean_Energy, SplitRatio = 0.8)

dados_treino <- subset(df2, dados_split == TRUE)
dados_teste <- subset(df2, dados_split == FALSE)

View(dados_treino)


# Aplicar machine learning 

# Retirando algumas variáveis menos importantes

dados_treino_1 <- dados_treino %>%
  select(-Car_name,-Model,-Brakes,-Num_doors,-Battery_capacity,-Num_seats)

dados_teste_1 <- dados_teste %>%
  select(-Car_name,-Model,-Brakes,-Num_doors,-Battery_capacity,-Num_seats)

View(dados_treino_1)

# Criação Modelo 1 Random Forest


model_random_forest_1 <- randomForest(Mean_Energy  ~ ., data=dados_treino_1, ntree=100,
                         importance=TRUE, type= "regression",mtry = 7)

# resumo
summary(model_random_forest_1)

# Realizando a predição
predicao1 <- predict(model_random_forest_1, newdata = dados_teste_1)

# Salvando em um Data Frame

predicao1 <- as.data.frame(predicao1)
View(predicao1)

avaliar <- cbind(dados_teste_1$Mean_Energy, predicao1$predicao1)

avaliar <- as.data.frame(avaliar)
colnames(avaliar) <- c('Observado', 'Previsto')

View(avaliar)

avaliar_diferenca <- avaliar[,1] - avaliar[,2]
avaliar$diferenca <- avaliar_diferenca

# MSE
mean((avaliar$diferenca)^2)

# RMSE

sqrt(mean((avaliar$diferenca)^2))

# Cálculo sse
sse1 <- sum((avaliar[,2] - dados_teste_1$Mean_Energy)^2)
sse1


# Cálculo ssr
ssr1 <- sum((avaliar[,2] - mean(dados_teste_1$Mean_Energy))^2)
ssr1



# Cálculo sst
sst1 <- ssr1 + sse1
sst1


# R-squared = 0,940

r_squared_1 = ssr1 / sst1
r_squared_1

?gbm

# Criação do modelo 2

model_gbm_2 = gbm(dados_treino_1$Mean_Energy ~.,
                data = dados_treino_1,
                distribution = "gaussian",
                cv.folds = 10,
                shrinkage = .01,
                n.minobsinnode = 1,
                n.trees = 500)

print(model_gbm2)

summary(model_gbm_2)




# Salvando a árvore de melhor performance

bestTreeForPrediction_2 = gbm.perf(model_gbm_2)

predicao_2 = predict.gbm(model_gbm_2, dados_teste_1,n.trees = bestTreeForPrediction_2)



predicao_2 <- as.data.frame(predicao_2)
View(predicao_2)

avaliar_2 <- cbind(dados_teste_1$Mean_Energy, predicao_2$predicao_2)

avaliar_2 <- as.data.frame(avaliar_2)
colnames(avaliar_2) <- c('Observado', 'Previsto')

View(avaliar_2)

avaliar_diferenca_2 <- avaliar_2[,1] - avaliar_2[,2]
avaliar_2$diferenca <- avaliar_diferenca_2

# MSE
mean((avaliar_2$diferenca)^2)

# RMSE

sqrt(mean((avaliar_2$diferenca)^2))

# Cálculo sse
sse2 <- sum((avaliar_2[,2] - dados_teste_1$Mean_Energy)^2)
sse2



# Cálculo ssr
ssr2 <- sum((avaliar_2[,2] - mean(dados_teste_1$Mean_Energy))^2)
ssr2



# Cálculo sst
sst2 <- ssr2 + sse2
sst2


# R-squared = 0,821

r_squared_2 = ssr2 / sst2
r_squared_2


model_gbm_3 = gbm(dados_treino_1$Mean_Energy ~. -Make,
                data = dados_treino_1,
                distribution = "gaussian",
                cv.folds = 5,
                shrinkage = .01,
                n.minobsinnode = 1,
                n.trees = 500)

?gbm


summary(model_gbm_3)


bestTreeForPrediction_3 = gbm.perf(model_gbm_3)


predicao_3 = predict.gbm(model_gbm_3, dados_teste_1,n.trees = bestTreeForPrediction_3)



predicao_3 <- as.data.frame(predicao_3)
View(predicao_3)

avaliar_3 <- cbind(dados_teste_1$Mean_Energy, predicao_3$predicao_3)

avaliar_3 <- as.data.frame(avaliar_3)
colnames(avaliar_3) <- c('Observado', 'Previsto')

View(avaliar_3)

avaliar_diferenca_3 <- avaliar_3[,1] - avaliar_3[,2]
avaliar_3$diferenca <- avaliar_diferenca_3

# MSE
mean((avaliar_3$diferenca)^2)

# RMSE

sqrt(mean((avaliar_3$diferenca)^2))

# Cálculo sse
sse3 <- sum((avaliar_3[,2] - dados_teste_1$Mean_Energy)^2)
sse3



# Cálculo ssr
ssr3 <- sum((avaliar_3[,2] - mean(dados_teste_1$Mean_Energy))^2)
ssr3



# Cálculo sst
sst3 <- ssr3 + sse3
sst3


# R-squared 

r_squared_3 = ssr3 / sst3
r_squared_3



cbind(r_squared_1,r_squared_2,r_squared_3)



# Max_dc_charge
# Max_torque
# Max_speed
# Min_price



model_gbm_4 = gbm(dados_treino_1$Mean_Energy ~. -Make -Max_dc_charge -Max_torque -Max_speed -Min_price,
                  data = dados_treino_1,
                  distribution = "gaussian",
                  cv.folds = 5,
                  shrinkage = .01,
                  n.minobsinnode = 1,
                  n.trees = 500)

View(dados_treino_1)

summary(model_gbm_4)


bestTreeForPrediction_4 = gbm.perf(model_gbm_4)


predicao_4 = predict.gbm(model_gbm_4, dados_teste_1,n.trees = bestTreeForPrediction_4)



predicao_4 <- as.data.frame(predicao_4)
View(predicao_4)

avaliar_4 <- cbind(dados_teste_1$Mean_Energy, predicao_4$predicao_4)

avaliar_4 <- as.data.frame(avaliar_4)
colnames(avaliar_4) <- c('Observado', 'Previsto')

View(avaliar_4)

avaliar_diferenca_4 <- avaliar_4[,1] - avaliar_4[,2]
avaliar_4$diferenca <- avaliar_diferenca_4

# MSE
mean((avaliar_4$diferenca)^2)

# RMSE

sqrt(mean((avaliar_4$diferenca)^2))

# Cálculo sse
sse4 <- sum((avaliar_4[,2] - dados_teste_1$Mean_Energy)^2)
sse4



# Cálculo ssr
ssr4 <- sum((avaliar_4[,2] - mean(dados_teste_1$Mean_Energy))^2)
ssr4



# Cálculo sst
sst4 <- ssr4 + sse4
sst4


# R-squared = 0,937

r_squared_4 = ssr4 / sst4
r_squared_4



cbind(r_squared_1,r_squared_2,r_squared_3,r_squared_4)





View(dados_treino_1)


# Modelo 5: Será igual ao modelo 4, mas excluindo outlier


boxplot(data_no_outlier$Wheelbase, main = "Wheelbase")
boxplot(data_no_outlier$Engine_power, main = "Engine_power")

boxplot(df$Wheelbase, main = "Wheelbase")
boxplot(df$Engine_power, main = "Engine_power")



quartiles <- quantile(df$Wheelbase, probs=c(.25, .75), na.rm = FALSE)
IQR <- IQR(df$Wheelbase)


Lower <- quartiles[1] - 1.5*IQR
Upper <- quartiles[2] + 1.5*IQR 

data_no_outlier <- subset(df, df$Wheelbase > Lower & df$Wheelbase < Upper)

dim(data_no_outlier)
dim(df)

View(data_no_outlier)




quartiles <- quantile(df$Engine_power, probs=c(.25, .75), na.rm = FALSE)
IQR <- IQR(df$Engine_power)


Lower <- quartiles[1] - 1.5*IQR
Upper <- quartiles[2] + 1.5*IQR 

data_no_outlier <- subset(data_no_outlier, data_no_outlier$Engine_power > Lower & data_no_outlier$Engine_power < Upper)

dim(data_no_outlier)
dim(df)

View(data_no_outlier)



df3 <- data_no_outlier %>%
  mutate(across(-c(1,2,3,7,8,18,19,20,25), minmax))
View(df3)



dados_split2 <- sample.split(df3$Mean_Energy, SplitRatio = 0.8)

dados_treino2 <- subset(df3, dados_split2 == TRUE)
dados_teste2 <- subset(df3, dados_split2 == FALSE)

View(dados_treino_2)


dados_treino_2 <- dados_treino2 %>%
  select(-Car_name,-Model,-Brakes,-Num_doors,-Battery_capacity,-Num_seats,-Make, -Max_dc_charge, -Max_torque, -Max_speed, -Min_price)

dados_teste_2 <- dados_teste2 %>%
  select(-Car_name,-Model,-Brakes,-Num_doors,-Battery_capacity,-Num_seats,-Make, -Max_dc_charge, -Max_torque, -Max_speed, -Min_price)



model_gbm_5 = gbm(dados_treino_2$Mean_Energy ~.,
                  data = dados_treino_2,
                  distribution = "gaussian",
                  cv.folds = 15,
                  shrinkage = .01,
                  n.minobsinnode = 1,
                  n.trees = 500)

?gbm


summary(model_gbm_5)


bestTreeForPrediction_5 = gbm.perf(model_gbm_5)


predicao_5 = predict.gbm(model_gbm_5,dados_teste_2,n.trees = bestTreeForPrediction_5)



View(dados_teste_2)

predicao_5 <- as.data.frame(predicao_5)
View(predicao_5)

avaliar_5 <- cbind(dados_teste_2$Mean_Energy, predicao_5$predicao_5)

avaliar_5 <- as.data.frame(avaliar_5)
colnames(avaliar_5) <- c('Observado', 'Previsto')

View(avaliar_5)

avaliar_diferenca_5 <- avaliar_5[,1] - avaliar_5[,2]
avaliar_5$diferenca <- avaliar_diferenca_5

# MSE
mean((avaliar_5$diferenca)^2)

# RMSE

sqrt(mean((avaliar_5$diferenca)^2))

# Cálculo sse
sse5 <- sum((avaliar_5[,2] - dados_teste_2$Mean_Energy)^2)
sse5



# Cálculo ssr
ssr5 <- sum((avaliar_5[,2] - mean(dados_teste_2$Mean_Energy))^2)
ssr5



# Cálculo sst
sst5 <- ssr4 + sse4
sst5


# R-squared = 0,937

r_squared_5 = ssr4 / sst4
r_squared_5


#RESULTADOS

cbind(r_squared_1,r_squared_2,r_squared_3,r_squared_4,r_squared_5)


# Conclusão:

# o modelo 1 (Random Forest) foi o melhor
# apesar que todos foram bons, acima de 80% da variância explicada  



