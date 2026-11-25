# Part 0: setup

# Rice variety classification and clustering project
# The goal is to study whether morphological measurements can distinguish
# Cammeo and Osmancik rice grains.
# First, the dataset is inspected through summary statistics and plots.
# Then, supervised models such as logistic regression, ridge, lasso,
# decision tree, random forest and KNN are compared for classification.
# Finally, PCA and k-means are used to check whether the data show a natural group structure.
# Dataset: UCI Machine Learning Repository - Rice (Cammeo and Osmancik)
# https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik

library(tidyverse)
library(corrplot)
library(sandwich)
library(lmtest)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)

rice = read.csv("Rice_Cammeo_Osmancik.csv")
rice$Class = as.factor(rice$Class)

head(rice)
str(rice)
dim(rice)
names(rice)

predictors = c("Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length",
               "Eccentricity", "Convex_Area", "Extent")

rice.num = rice[, predictors]


#descriptive analysis

colSums(is.na(rice))
sum(duplicated(rice))
table(rice$Class)
prop.table(table(rice$Class)) * 100


# class plot

p.class = ggplot(rice, aes(x = Class)) +
  geom_bar() +
  labs(title = "Class Distribution", x = "Rice Variety", y = "Count") +
  theme_minimal()
p.class


summary(rice.num)


mean.by.class = aggregate(rice.num, by = list(Class = rice$Class), FUN = mean)
mean.by.class


#corrplot

cor.table = cor(rice.num)
cor.table

corrplot(cor.table, type = "upper",
         tl.col = "black", tl.srt = 45)


#boxplots

boxplot(rice$Area, main = "Area")
boxplot(rice$Perimeter, main = "Perimeter")
boxplot(rice$Major_Axis_Length, main = "Major Axis Length")
boxplot(rice$Minor_Axis_Length, main = "Minor Axis Length")
boxplot(rice$Eccentricity, main = "Eccentricity")
boxplot(rice$Convex_Area, main = "Convex Area")
boxplot(rice$Extent, main = "Extent")


#density plots

p.area = ggplot(rice, aes(x = Area, fill = Class)) +
  geom_density(alpha = 0.5) +
  labs(title = "Area Distribution", x = "Area", y = "Density") +
  theme_minimal()
p.area

p.perimeter = ggplot(rice, aes(x = Perimeter, fill = Class)) +
  geom_density(alpha = 0.5) +
  labs(title = "Perimeter Distribution", x = "Perimeter", y = "Density") +
  theme_minimal()
p.perimeter

p.major = ggplot(rice, aes(x = Major_Axis_Length, fill = Class)) +
  geom_density(alpha = 0.5) +
  labs(title = "Major Axis Length Distribution", x = "Major Axis Length", y = "Density") +
  theme_minimal()
p.major

p.convex = ggplot(rice, aes(x = Convex_Area, fill = Class)) +
  geom_density(alpha = 0.5) +
  labs(title = "Convex Area Distribution", x = "Convex Area", y = "Density") +
  theme_minimal()
p.convex


#scatter plot

p.scatter = ggplot(rice, aes(x = Area, y = Major_Axis_Length, color = Class)) +
  geom_point(alpha = 0.5) +
  labs(title = "Area and Major Axis Length", x = "Area", y = "Major Axis Length") +
  theme_minimal()
p.scatter

# Part 2: supervised learning

# target

rice$target.var = ifelse(rice$Class == "Cammeo", 1, 0)
table(rice$Class, rice$target.var)


# split

set.seed(123)

train.index = sample(1:nrow(rice), size = 0.5 * nrow(rice))

train = rice[train.index, ]
test = rice[-train.index, ]

dim(train)
dim(test)

prop.table(table(train$Class)) * 100
prop.table(table(test$Class)) * 100


# variables

x.train = train[, predictors]
x.test = test[, predictors]

y.train = train$target.var
y.test = test$target.var

# standardization


train.mean = apply(x.train, 2, mean)
train.sd = apply(x.train, 2, sd)

x.train.scaled = scale(x.train, center = train.mean, scale = train.sd)
x.test.scaled = scale(x.test, center = train.mean, scale = train.sd)


# functions

mse = function(model.prob, real.values) {
  errors = model.prob - real.values
  squared.errors = errors^2
  mse.value = mean(squared.errors)
  return(mse.value)
}

accuracy = function(model.classification, real.values) {
  correct.predictions = model.classification == real.values
  accuracy.value = mean(correct.predictions)
  return(accuracy.value)
}

conf.matrix = function(model.classification, real.values) {
  matrix.result = table(Predicted = model.classification, Real.values = real.values)
  return(matrix.result)
}


# ols

ols.model = lm(target.var ~ Area + Perimeter + Major_Axis_Length + Minor_Axis_Length +
                 Eccentricity + Convex_Area + Extent, data = train)
summary(ols.model)

ols.prob = predict(ols.model, newdata = test)

range(ols.prob)

ols.classification = ifelse(ols.prob >= 0.5, 1, 0)

ols.mse = mse(ols.prob, y.test)
ols.accuracy = accuracy(ols.classification, y.test)
ols.conf.matrix = conf.matrix(ols.classification, y.test)

ols.mse
ols.accuracy
ols.conf.matrix

ols.robust = coeftest(ols.model, vcov = vcovHC(ols.model, type = "HC1"))
ols.robust


# logistic regression

logit.model = glm(target.var ~ Area + Perimeter + Major_Axis_Length + Minor_Axis_Length +
                    Eccentricity + Convex_Area + Extent, data = train, family = binomial)
summary(logit.model)

logit.prob = predict(logit.model, newdata = test, type = "response")
logit.classification = ifelse(logit.prob >= 0.5, 1, 0)

logit.mse = mse(logit.prob, y.test)
logit.accuracy = accuracy(logit.classification, y.test)
logit.conf.matrix = conf.matrix(logit.classification, y.test)

logit.mse
logit.accuracy
logit.conf.matrix


# stepwise logistic

step.logit.model = step(logit.model, direction = "both")
summary(step.logit.model)

step.logit.model$anova 


# ridge

x.train.matrix = as.matrix(x.train.scaled)
x.test.matrix = as.matrix(x.test.scaled)

set.seed(123)

ridge.model = cv.glmnet(x.train.matrix, y.train, family = "binomial", alpha = 0)
plot(ridge.model)

ridge.lambda = ridge.model$lambda.min
ridge.lambda 

ridge.prob = predict(ridge.model, newx = x.test.matrix,
                     s = ridge.lambda, type = "response")
ridge.prob = as.vector(ridge.prob) 

ridge.classification = ifelse(ridge.prob >= 0.5, 1, 0)

ridge.mse = mse(ridge.prob, y.test)
ridge.accuracy = accuracy(ridge.classification, y.test)
ridge.conf.matrix = conf.matrix(ridge.classification, y.test)

ridge.mse
ridge.accuracy
ridge.conf.matrix

coef(ridge.model, s = ridge.lambda) 


# lasso

set.seed(123)

lasso.model = cv.glmnet(x.train.matrix, y.train, family = "binomial", alpha = 1)
plot(lasso.model)

lasso.lambda = lasso.model$lambda.min
lasso.lambda 

lasso.prob = predict(lasso.model, newx = x.test.matrix,
                     s = lasso.lambda, type = "response")
lasso.prob = as.vector(lasso.prob)

lasso.classification = ifelse(lasso.prob >= 0.5, 1, 0)

lasso.mse = mse(lasso.prob, y.test)
lasso.accuracy = accuracy(lasso.classification, y.test)
lasso.conf.matrix = conf.matrix(lasso.classification, y.test)

lasso.mse
lasso.accuracy
lasso.conf.matrix

coef(lasso.model, s = lasso.lambda) 


# tree

tree.train = train[, c(predictors, "Class")]
tree.test = test[, c(predictors, "Class")]

tree.model = rpart(Class ~ ., data = tree.train, method = "class")
printcp(tree.model) 

rpart.plot(tree.model)

tree.prob = predict(tree.model, newdata = tree.test, type = "prob")[, "Cammeo"]
tree.classification = predict(tree.model, newdata = tree.test, type = "class")
tree.classification = ifelse(tree.classification == "Cammeo", 1, 0)

tree.mse = mse(tree.prob, y.test)
tree.accuracy = accuracy(tree.classification, y.test)
tree.conf.matrix = conf.matrix(tree.classification, y.test)

tree.mse
tree.accuracy
tree.conf.matrix


# random forest

set.seed(123)

rf.model = randomForest(Class ~ ., data = tree.train, ntree = 500, mtry = 2, importance = TRUE)
rf.model

rf.prob = predict(rf.model, newdata = tree.test, type = "prob")[, "Cammeo"]
rf.classification = predict(rf.model, newdata = tree.test)
rf.classification = ifelse(rf.classification == "Cammeo", 1, 0)

rf.mse = mse(rf.prob, y.test)
rf.accuracy = accuracy(rf.classification, y.test)
rf.conf.matrix = conf.matrix(rf.classification, y.test)

rf.mse
rf.accuracy
rf.conf.matrix

varImpPlot(rf.model) 


# knn

set.seed(123)

k.values = 1:50
knn.accuracy.values = rep(0, length(k.values))

for (i in 1:length(k.values)) {
  knn.model = knn(train = x.train.scaled,
                  test = x.test.scaled,
                  cl = train$Class,
                  k = k.values[i])
  
  knn.accuracy.values[i] = mean(knn.model == test$Class)
}

knn.results = data.frame(K = k.values, Accuracy = knn.accuracy.values)
knn.results$Test.error = 1 - knn.results$Accuracy

knn.results

best.k = knn.results$K[which.max(knn.results$Accuracy)]
best.k 

p.knn = ggplot(knn.results, aes(x = K, y = Test.error)) +
  geom_line() +
  geom_point() +
  labs(title = "KNN Test Error", x = "K", y = "Test Error") +
  theme_minimal()
p.knn

knn.final = knn(train = x.train.scaled,
                test = x.test.scaled,
                cl = train$Class,
                k = best.k)

knn.classification = ifelse(knn.final == "Cammeo", 1, 0)

knn.accuracy = accuracy(knn.classification, y.test)
knn.conf.matrix = conf.matrix(knn.classification, y.test)

knn.accuracy
knn.conf.matrix


# results

model.names = c("OLS", "Logistic Regression", "Ridge Logistic", "Lasso Logistic",
                "Decision Tree", "Random Forest", "KNN")

model.accuracy = c(ols.accuracy, logit.accuracy, ridge.accuracy, lasso.accuracy,
                   tree.accuracy, rf.accuracy, knn.accuracy)

model.correct = c(sum(ols.classification == y.test),
                  sum(logit.classification == y.test),
                  sum(ridge.classification == y.test),
                  sum(lasso.classification == y.test),
                  sum(tree.classification == y.test),
                  sum(rf.classification == y.test),
                  sum(knn.classification == y.test))

model.errors = length(y.test) - model.correct

supervised.results = data.frame(model.names, model.accuracy, model.errors)
names(supervised.results) = c("Model", "Accuracy", "Errors")

supervised.results

p.models = ggplot(supervised.results, aes(x = Model, y = Accuracy)) +
  geom_col() +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()
p.models

# Part 3: unsupervised learning

# pca

pca.model = prcomp(rice.num, center = TRUE, scale. = TRUE)
summary(pca.model)

pca.variance = pca.model$sdev^2
pca.explained = pca.variance / sum(pca.variance)
pca.cumulative = cumsum(pca.explained)

pca.results = data.frame(
  Component = paste0("PC", 1:length(pca.explained)),
  Explained = pca.explained,
  Cumulative = pca.cumulative
)

pca.results

plot(pca.explained, type = "b",
     main = "PCA Explained Variance",
     xlab = "Principal Component",
     ylab = "Proportion of Variance Explained")

plot(pca.cumulative, type = "b",
     main = "PCA Cumulative Variance",
     xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained")

pca.plot.data = data.frame(
  PC1 = pca.model$x[, 1],
  PC2 = pca.model$x[, 2],
  Class = rice$Class
)

p.pca = ggplot(pca.plot.data, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(alpha = 0.5) +
  labs(title = "PCA Plot", x = "PC1", y = "PC2") +
  theme_minimal()
p.pca


#k-means on scaled predictors

set.seed(123)

kmeans.data = scale(rice.num)

kmeans.model = kmeans(kmeans.data, centers = 2, nstart = 10)
kmeans.model

table.predictors = table(kmeans.model$cluster, rice$Class)
table.predictors

kmeans.plot.data = data.frame(
  PC1 = pca.model$x[, 1],
  PC2 = pca.model$x[, 2],
  Cluster = as.factor(kmeans.model$cluster),
  Class = rice$Class
)

p.kmeans = ggplot(kmeans.plot.data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "K-means on Scaled Predictors", x = "PC1", y = "PC2") +
  theme_minimal()
p.kmeans


#k-means on pca components

pca.data = pca.model$x[, 1:2]

set.seed(123)

kmeans.pca.model = kmeans(pca.data, centers = 2, nstart = 10)
kmeans.pca.model

table.pca = table(kmeans.pca.model$cluster, rice$Class)
table.pca

kmeans.pca.plot.data = data.frame(
  PC1 = pca.data[, 1],
  PC2 = pca.data[, 2],
  Cluster = as.factor(kmeans.pca.model$cluster),
  Class = rice$Class
)

p.kmeans.pca = ggplot(kmeans.pca.plot.data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "K-means on PCA Components", x = "PC1", y = "PC2") +
  theme_minimal()
p.kmeans.pca


#results

correct.predictors = max(table.predictors[1, ]) + max(table.predictors[2, ])
accuracy.predictors = correct.predictors / nrow(rice)
errors.predictors = nrow(rice) - correct.predictors

correct.pca = max(table.pca[1, ]) + max(table.pca[2, ])
accuracy.pca = correct.pca / nrow(rice)
errors.pca = nrow(rice) - correct.pca

unsupervised.results = data.frame(
  Method = c("K-means on scaled predictors", "K-means on PCA components"),
  Accuracy = c(accuracy.predictors, accuracy.pca),
  Errors = c(errors.predictors, errors.pca)
)

unsupervised.results

p.unsupervised = ggplot(unsupervised.results, aes(x = Method, y = Accuracy)) +
  geom_col() +
  labs(title = "Unsupervised Results", x = "Method", y = "Accuracy") +
  theme_minimal()
p.unsupervised




