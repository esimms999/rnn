
max_words <- 10000
max_len <- 100

# Load the IMDB data
imdb <- dataset_imdb(num_words = max_words)

# Split the data into training and test sets
x_train <- imdb$train$x
y_train <- imdb$train$y
x_test <- imdb$test$x
y_test <- imdb$test$y

# Pad the sequences to have a fixed length
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test <- pad_sequences(x_test, maxlen = max_len)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = 32) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

scores <- model %>% evaluate(x_test, y_test,
                             verbose = 0)
print(paste("Test accuracy:", scores[[2]]))
