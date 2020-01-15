import models.nn


MODELS = {
    "baseline": lambda *args, **kwargs: models.nn.Baseline(),
    "sven": lambda *args, **kwargs: models.nn.Sven(
        dict(
            conv3d_num_filters=64,
            conv3d_kernel_size=(3, 5, 5),
            encoder_rnn_num_layers=1,
            encoder_rnn_dropout=0.0,
        )
    ),
    "sven-generic": lambda params: models.nn.Sven(params),
}
