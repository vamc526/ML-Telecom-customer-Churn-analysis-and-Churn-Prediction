Flags <- flags(
  flag_numeric("nodes", 128),
  flag_numeric("nodes2", 32),
  flag_numeric("batch_size", 50),
  flag_string("activation", "relu"),
  flag_string("activation2", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 30),
  flag_numeric("dropout", 0.2),
  flag_numeric("dropout2", 0.2)
  
)

callbacks = list(callback_early_stopping(monitor = "val_loss",patience = 10, restore_best_weights = TRUE))


model = keras_model_sequential()
model %>%
  layer_dense(units = Flags$nodes, activation = Flags$activation,input_shape = dim(x_train)[2]) %>%
  layer_dropout(Flags$dropout) %>%
  layer_dense(units = Flags$nodes2, activation = Flags$activation2) %>%
  layer_dropout(Flags$dropout2) %>%
  layer_dense(units = 1, activation = "sigmoid")


model %>% compile(optimizer = optimizer_adam(lr = Flags$learning_rate), loss = "binary_crossentropy", metrics = "acc")

model %>% fit(
  as.matrix(x_train), y_train, epochs = Flags$epochs, batch_size = Flags$batch_size, validation_data=list(as.matrix(x_test), y_test),class_weight = list("0"=w_no, "1"=w_yes),callbacks=callbacks)
