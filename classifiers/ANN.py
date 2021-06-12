from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(  # -----The architecture:------#
    activation="relu",  # What is the activation function between neurons {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}?
    hidden_layer_sizes=(100, ),  # What is the architecture? what happens if we add more layers?
    alpha=0.035,  # The regularization: loss + alpha*W^2, you know it as lambda.
    # -----The optimizer:------#
    solver="adam",  # Stochastic Gradient Descent, other optimizers are out of the scope of the course.
    learning_rate_init=0.0015,  # What is the initial learning rate? in some optimizers the learning rate changes.

    learning_rate="invscaling",  # How does the learning rate update itself? {‘constant’, ‘invscaling’, ‘adaptive’}
    power_t=0.5,  # When we choose learning rate to be invscaling, it means that we multiply this number each epoch.
    early_stopping=False,
    # If True, then we set an internal validation data and stop training when there is no imporovement.
    tol=1e-4,  # A broad concept of converges, when we can say the algorithm converged?
    beta_1=0.87,
    batch_size=40,  # The number of samples each batch.
    max_iter=1500,  # The total number of epochs.
    warm_start=False,  # if we fit at the second time, do we start from the last fit?

    random_state=42  # seed
)



# ann = MLPClassifier(  # -----The architecture:------#
#     activation="relu",  # What is the activation function between neurons {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}?
#     hidden_layer_sizes=(100,),  # What is the architecture? what happens if we add more layers?
#     alpha=0.035,  # The regularization: loss + alpha*W^2, you know it as lambda.
#     # -----The optimizer:------#
#     solver="adam",  # Stochastic Gradient Descent, other optimizers are out of the scope of the course.
#     learning_rate_init=0.0015,  # What is the initial learning rate? in some optimizers the learning rate changes.
#     learning_rate="invscaling",  # How does the learning rate update itself? {‘constant’, ‘invscaling’, ‘adaptive’}
#     power_t=0.5,  # When we choose learning rate to be invscaling, it means that we multiply this number each epoch.
#     early_stopping=False,
#     # If True, then we set an internal validation data and stop training when there is no imporovement.
#     tol=1e-4,  # A broad concept of converges, when we can say the algorithm converged?
#     batch_size=40,
#     max_iter=300,  # The total number of epochs.
#     warm_start=False,
#     beta_1=0.87, # if we fit at the second time, do we start from the last fit?
#     random_state=42# seed
# )