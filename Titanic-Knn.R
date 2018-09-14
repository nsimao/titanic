# Configuração do diretório
#setwd("~/Nelson/DS/Projetos/Titanic")

# Leitura dos dados de treino
dados_treino <- read.csv('Dados/train.csv')
dados_teste <- read.csv('Dados/test.csv')

# Remoção de Colunas
library("dplyr")
dados_treino <- select(dados_treino, -PassengerId, -Name, -Ticket, -Cabin, -Embarked)

# Analise exploratoria de dados

# Visualizando os dados
library(ggplot2)
ggplot(dados_treino,aes(Survived)) + geom_bar()
ggplot(dados_treino,aes(Pclass)) + geom_bar(aes(fill = factor(Pclass)), alpha = 0.5)
ggplot(dados_treino,aes(Sex)) + geom_bar(aes(fill = factor(Sex)), alpha = 0.5)
ggplot(dados_treino,aes(Age)) + geom_histogram(fill = 'blue', bins = 20, alpha = 0.5)
ggplot(dados_treino,aes(SibSp)) + geom_bar(fill = 'red', alpha = 0.5)
ggplot(dados_treino,aes(Fare)) + geom_histogram(fill = 'green', color = 'black', alpha = 0.5)

# Limpando os dados
# Para tratar os dados missing, usaremos o recurso de imputation, ou seja, vamos preencher os dados missing,
# com o valor da media dos dados
pl <- ggplot(dados_treino, aes(Pclass,Age)) + geom_boxplot(aes(group = Pclass, fill = factor(Pclass), alpha = 0.4)) 
pl + scale_y_continuous(breaks = seq(min(0), max(80), by = 2))

# group_by()
teste <-  as.data.frame(dados_treino %>% 
                        group_by(Sex, Survived) %>%
                        summarise(total = n()) %>% 
                        mutate(label = paste(as.character(Sex), as.character(Survived))))

ggplot(teste, aes(label,total)) + geom_bar(stat = "identity", aes(fill = factor(label), alpha = 0.4))


dfIdade <- dados_treino
dfIdade_media_age <- mean(dfIdade$Age, na.rm = TRUE)
dfIdade$Age[is.na(dfIdade$Age)] <- dfIdade_media_age

dfIdade$Age <- ifelse(dfIdade$Age<18, 1, 0)

any(is.na(dfIdade))

teste2 <-  as.data.frame(dfIdade %>% 
                          group_by(Age, Sex, Survived) %>%
                          summarise(total = n()) %>% 
                          mutate(label = paste(as.character(ifelse(Age==1, "Criança", "Adulto")), 
                                               as.character(Sex), 
                                               ifelse(Survived==1, "Sobreviveu", "Morreu"))) %>%
                           arrange(label)
                         )

ggplot(teste2, aes(label,total)) + geom_bar(stat = "identity", aes(fill = factor(label), alpha = 0.4))


# Verificando se existem dados NA no DataFrame
any(is.na(dados_treino))
any(is.na(dados_teste))

# Utilizando o pacote Amelia para visualizar a distribuição de dados NA no DataFrame
#install.packages("Amelia")
library(Amelia)
missmap(dados_treino, main = "Titanic Dados Treino - Mapa de Dados Missing", 
        col = c("red", "white"))

missmap(dados_teste, main = "Titanic Dados Teste - Mapa de Dados Missing", 
        col = c("red", "white"))

# Preenchendo dados Missing
treino_media_age <- mean(dados_treino$Age, na.rm = TRUE)
dados_treino$Age[is.na(dados_treino$Age)] <- treino_media_age

teste_media_age <- mean(dados_teste$Age, na.rm = TRUE)
dados_teste$Age[is.na(dados_teste$Age)] <- teste_media_age

teste_media_fare <- mean(dados_teste$Fare, na.rm = TRUE)
dados_teste$Fare[is.na(dados_teste$Fare)] <- teste_media_fare

# Alterando a idade 0 = Adultos (>=18), 1 = Crianças (<18)
dados_treino$Age <- ifelse(dados_treino$Age<18, 1, 0)
dados_teste$Age <- ifelse(dados_teste$Age<18, 1, 0)

# Alterando Sexo 0 = male, 1 = female
dados_treino$Sex <- sapply(as.character(dados_treino$Sex), switch, 'male' = 0, 'female' = 1)
dados_teste$Sex <- sapply(as.character(dados_teste$Sex), switch, 'male' = 0, 'female' = 1)


normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalização dos dados
dados_treino$Fare = normalizar(dados_treino$Fare)
dados_teste$Fare = normalizar(dados_teste$Fare)

# Filtrando as colunas numéricas para correlação
colunas_numericas <- sapply(dados_treino, is.numeric)
colunas_numericas
data_cor <- cor(dados_treino[,colunas_numericas])

# Pacotes para visualizar a análise de correlação
# install.packages('corrplot')
library(corrplot)

# Criando um corrplot
corrplot(data_cor, method = 'color')

# install.packages("caTools")
library(caTools)

# Criando as amostras de forma randômica
?sample.split
amostra <- sample.split(dados_treino$Survived, SplitRatio = 0.70)

# Criando dados de treino - 70% dos dados
treino = subset(dados_treino, amostra == TRUE)

# Criando dados de teste - 30% dos dados
teste = subset(dados_treino, amostra == FALSE)

treino_labels <- treino[, 1] 
teste_labels <- teste[, 1]


# Testando o modelo com os dados de treino e testes
library(class)

prev = NULL
taxa_erro = NULL

for(i in 1:20){
  prev = knn(train = treino, test = teste, cl = treino_labels, k = i)
  taxa_erro[i] = sum(teste_labels != prev)
}

# Obtendo os valores de k e das taxas de erro
library(ggplot2)
k.values <- 1:20
df_erro <- data.frame(taxa_erro, k.values)
df_erro

# Visualizando a taxa de erro e k
ggplot(df_erro, aes(x = k.values, y = taxa_erro)) + geom_point()+ geom_line(lty = "dotted", color = 'red')


# Seleciono k possivelmente ideal e gerando dataframe para submissão
survived <- dados_treino$Survived
passengers <- dados_teste$PassengerId
dados_teste <- select(dados_teste, -PassengerId, -Name, -Ticket, -Cabin, -Embarked)
dados_treino_final <- select(dados_treino, -survived)

prev = knn(train = dados_treino_final, test = dados_teste, cl = survived, k = 2)
submission <- data.frame(PassengerId = passengers,Survived = prev)
write.csv(submission,'titanic_previsoes.csv', row.names = FALSE, quote = FALSE)


