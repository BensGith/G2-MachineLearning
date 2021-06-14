from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(  # -----The architecture:------#
    activation="relu",  # What is the activation function between neurons {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}?
    hidden_layer_sizes=(100,),  # What is the architecture? what happens if we add more layers?
    alpha=0.035,  # The regularization: loss + alpha*W^2, you know it as lambda.
    # -----The optimizer:------#
    solver="adam",  # Adam Solver
    learning_rate_init=0.0015,  # What is the initial learning rate? in some optimizers the learning rate changes.
    learning_rate="invscaling",  # How does the learning rate update itself? {‘constant’, ‘invscaling’, ‘adaptive’}
    power_t=0.5,  # When we choose learning rate to be invscaling, it means that we multiply this number each epoch.
    early_stopping=False,
    # If True, then we set an internal validation data and stop training when there is no imporovement.
    tol=1e-4,
    batch_size=10,  # The number of samples each batch.
    max_iter=500,   # The total number of epochs.
    beta_1=0.8, # Parametar for adam solver
    warm_start=False,  # if we fit at the second time, do we start from the last fit?
    random_state=42  # seed

)