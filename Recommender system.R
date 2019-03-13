# STAT 642- DATA MINING PROJECT
# Recommendation System Collaborative Filtering

#Data: MovieLens Data 
# The dataset used was from MovieLens, and is publicly available at 
# http://grouplens.org/datasets/movielens/latest. 
# In order to keep the recommender simple, the smallest dataset available which contaied 
# 105339 ratings and 6138 tag applications across 10329 movies. The data are contained in four files: links.csv, 
# movies.csv, ratings.csv and tags.csv. Movies.csv and Ratings.csv were used to build a recommendation system.

setwd('C:/Users/Bharat/Desktop/Portfolium/Movie Recommendor system/ml-latest-small')

library(Matrix)
library(arules)
library(proxy)
library(ggplot2)
library(data.table)
library(reshape2)
library(recommenderlab)

# Load the data files
movies <- read.csv("movies.csv", stringsAsFactors=F)
str(movies)
ratings <- read.csv("ratings.csv", stringsAsFactors=T)
str(ratings)

#Note that UserID and MovieID are imported as numerical datatype but they should be treated as Factors
movid1 <- as.factor(movies$movieId)
movies$movieId <- movid1 

userr <- as.factor(ratings$userId)
movid <- as.factor(ratings$movieId)
ratings$userId <- userr
ratings$movieId <- movid

str(ratings)
# It can be noted that there are 671 users who has rated 9067 movies with rating scale ranging from 0.5 to 5
# with 5 being the max score.

# Data Pre-processing & Expoloratory Data Anlaysis
# Plotting the ratings across the scale to understand the count for each score and distribution
table(ratings$rating)
ggplot(data=ratings, aes(ratings$rating)) + geom_histogram(breaks=seq(0, 5, by = .5), col="red", aes(fill=..count..)) + scale_fill_gradient("Count", low = "green", high = "red")
# It looks like most of the ratings given are centered around score 4 and 3 (Left- tailes distribution)



# Model building

# Casting Utility matrix with users in row, movies in columnand their respective user/ movie ratings 
# as values in the matrix. It's well known that al the users watching movies will seldom rate movies which
# results in a sparse rating matrix. We can filter for the users who has rated atleast a minimal amount
# of movies and movies which are rated by some number of users that can reduce the complexity of dealing with 
# sparse matrix. Filtering condition --> (User = 10), (Movies = 5)


# Creating Utility (U) matrix ---> Rows as userID and Columns as movieID 
ratmatrix2 <- dcast(ratings, userId ~ movieId, value.var = "rating", na.rm=FALSE)
# Utility matrix is of dimension 671 (Users) x 9067 (Movies)

# Remove userIds
ratmatrix1 <- as.matrix(ratmatrix2[,-1]) 
View(ratmatrix1)

# Convert rating matrix into a recommenderlab sparse matrix which represents N/A missing values
# with dots
ratmat1 <- as(ratmatrix1, "realRatingMatrix")


# Extracting data that comprises of at least 10 ratings per user and 5 ratings per movie
utilmatrix <- ratmat1 [rowCounts(ratmat1) > 10, colCounts(ratmat1) > 5]
minrowcnt <- min(rowCounts(utilmatrix))
dim(utilmatrix)
# It can be seen that dimension of Utility matrix is reduced to 671(Users) x 3099(Movies)
# Sparsity of the matrix is reduced and it can be made sure that recommendation engine is
# built on data which is populated to improve the accuracy



# Splitting Training/ Test data

which_train <- sample(x = c(TRUE, FALSE), 
                      size = nrow(utilmatrix),
                      replace = TRUE, 
                      prob = c(0.8, 0.2))

recc_data_train <- utilmatrix[which_train, ]
recc_data_test <- utilmatrix[!which_train, ]


# Method: User Based Collaborative filtering
# Similarity Calculation Method: Centered Cosine Similarity/ Pearson Correlation coeffecient
# Nearest Neighbors: 10 (This parameter plays pivotal role in improving the accuracy measures of model)

# Normalizing the dataset so that missing values are not treated as zero. This helps in distributing the ratings
# around the mean of each user thus avoiding model bias
utilnorm <- normalize(utilmatrix)

# Image comparison for sparsity of matrix
image(ratmatrix1)
image(utilnorm)
# Sparsity of matrix has been reduced considerably

# Model training
ubmodel <- Recommender(utilnorm, method = "UBCF", param=list(method="cosine", n=15))

# Obtain top 10 recommendations for User no.1 
useridno <- 1 # Change the userid number to get recommendations for any particular user
ubpred <- predict (ubmodel, utilnorm[1], n=10)

# Convert recommenderlab object to readable list
ubpredlist <- as(ubpred, "list")
ubpredlist

# Obtain recommendations 
recom_result <- matrix(0,10)
for (i in c(1:10)){
  recom_result[i] <- movies[as.integer(ubpredlist[[1]][i]),2]
}
print(recom_result)

# Model evaluation
evascheme <- evaluationScheme (ratmat1, method= "cross-validation", k=5, given=3, goodRating=5) 
evaresults <- evaluate (evascheme, method="UBCF", n=c(1, 5, 8, 15, 20))
evaresults1 <- getConfusionMatrix(evaresults)[1]
print(evaresults1)

# ROC curve
# It estimates AUC (Area Under curve) which represents prediction accuracy of the model
plot(evaresults, main = "ROC curve", col= "blue", lty = 6, pch = "o")

# Precision vs Recall (Accuracy measures) plot
plot(evaresults, "prec/rec", main = "Precision vs Recall", col= "blue", lty = 6, pch = "x")

# Optimizing k (nearest neighborhood users to consider for making recommendations) value and understand 
# how it affects the accuracy  measures

vector_k <- c(5, 10, 15, 20, 25)
models_to_evaluate <- lapply(vector_k, function(k){
  list(name = "UBCF",
       param = list(method = "cosine", k = k))
})
names(models_to_evaluate) <- paste0("IBCF_k_", vector_k)

# Item- Based Collaborative filetring 
recommender_models <- recommenderRegistry$get_entries(dataType ="realRatingMatrix")
recommender_models$IBCF_realRatingMatrix$parameters

recc_model <- Recommender(data = recc_data_train, 
                          method = "IBCF",
                          parameter = list(k = 30))

recc_model
class(recc_model)

n_recommended <- 10 # the number of items to recommend to each user

recc_predicted <- predict(object = recc_model, 
                          newdata = recc_data_test, 
                          n = n_recommended)
recc_predicted

ibmodel <- Recommender(data = utilnorm, method = "IBCF", parameter = list(k = 20, method = "Cosine"))
ibmodel

ibmodel_details <- getModel(ibmodel)
# dgCMatrix Sparse matrix
class(ibmodel_details$sim)
dim(ibmodel_details$sim)

moviesug <- 1000
image(ibmodel_details$sim[1:moviesug, 1:moviesug],
      main = "Heatmap of the first 1000 rows and columns")


# Obtain top 10 movie recommendations to suggest it to the user
n_recommended <- 10 
recc_predicted <- predict(object= ratmatrix2, newdata = ibmodel, n = n_recommended)
recc_predicted


# Convert recommenderlab object to readable list
ibpredlist <- as(ubpred, "list")
ubpredlist

# Obtain recommendations

recc_user_1 <- recc_predicted@items[[1]] # recommendation for the first user
movies_user_1 <- recc_predicted@itemLabels[recc_user_1]
movies_user_2 <- movies_user_1
for (i in 1:10){
  movies_user_2[i] <- as.character(subset(movies, 
                                          movies$movieId == movies_user_1[i])$title)
}
movies_user_2

# Model evaluation
evascheme <- evaluationScheme (ratmat1, method= "cross-validation", k=5, given=3, goodRating=5) 
evaresults <- evaluate (evascheme, method="UBCF", n=c(1, 5, 8, 15, 20))
evaresults1 <- getConfusionMatrix(evaresults)[1]
print(evaresults1)





