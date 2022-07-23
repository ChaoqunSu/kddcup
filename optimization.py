import paddle as P


def get_optimizer(model, learning_rate):
    g_clip = P.nn.ClipGradByNorm(50.0)
    opt = P.optimizer.Adam(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        grad_clip=g_clip)
    return opt
