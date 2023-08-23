install.packages("keras")
install.packages("caret")

# read in the packages we'll use
library(keras) # for deep learning
library(tidyverse) # general utility functions
library(caret) # machine learning utility functions

# read in our data
weather_data <- read_csv("seattleWeather_1948-2017.csv")

# check out the first few rows
head(weather_data)

# set some parameters for our model
max_len <- 6 # the number of previous examples we'll look at
batch_size <- 32 # number of sequences to look at at one time during training
total_epochs <- 15 # how many times we'll look @ the whole dataset while training our model

# set a random seed for reproducability
set.seed(123)

# select out the colum with info on how often it rained
rain <- weather_data$RAIN

# summerize this
table(rain)

# Cut the text in overlapping sample sequences of max_len characters

# get a list of start indexes for our (overlapping) chunks
start_indexes <- seq(1, length(rain) - (max_len + 1), by = 3)

# create an empty matrix to store our data in
weather_matrix <- matrix(nrow = length(start_indexes), ncol = max_len + 1)

# fill our matrix with the overlapping slices of our dataset
for (i in 1:length(start_indexes)){
  weather_matrix[i,] <- rain[start_indexes[i]:(start_indexes[i] + max_len)]
}

# make sure it's numeric
weather_matrix <- weather_matrix * 1

# remove na's if you have them
if(anyNA(weather_matrix)){
  weather_matrix <- na.omit(weather_matrix)
}

# split our data into the day we're predict (y), and the 
# sequence of days leading up to it (X)
X <- weather_matrix[,-ncol(weather_matrix)]
y <- weather_matrix[,ncol(weather_matrix)]

# create an index to split our data into testing & training sets
training_index <- createDataPartition(y, p = .9, 
                                      list = FALSE, 
                                      times = 1)

# training data
X_train <- array(X[training_index,], dim = c(length(training_index), max_len, 1))
y_train <- y[training_index]

# testing data
X_test <- array(X[-training_index,], dim = c(length(y) - length(training_index), max_len, 1))
y_test <- y[-training_index]

# initialize our model
model <- keras_model_sequential()



