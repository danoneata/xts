import models.nn


MODELS = {
    "sven": lambda dataset_params, *args, **kwargs: models.nn.Sven(
        dataset_params,
        dict(
            conv3d_num_filters=64,
            conv3d_kernel_size=(3, 5, 5),
            encoder_rnn_num_layers=1,
            encoder_rnn_dropout=0.0,
        )
    ),
    "sven-generic": lambda dataset_params, params: models.nn.Sven(dataset_params, params),
    "magnus": lambda dataset_params, *args, **kwargs: models.nn.Sven(
        dataset_params,
        dict(
            conv3d_num_filters=128,
            conv3d_kernel_size=(5, 5, 5),
            encoder_rnn_num_layers=2,
            encoder_rnn_dropout=0.1,
        )
    ),
    "magnus-multi-speaker": lambda dataset_params, *args, **kwargs: models.nn.Sven(
        dataset_params,
        dict(
            conv3d_num_filters=128,
            conv3d_kernel_size=(5, 5, 5),
            encoder_rnn_num_layers=2,
            encoder_rnn_dropout=0.1,
            speaker_embedding_dim=32,
        )
    ),
    "bjorn": lambda dataset_params, *args, **kwargs: models.nn.Bjorn(
        dataset_params,
        dict(
            conv3d_num_filters=128,
            conv3d_kernel_size=(5, 5, 5),
            encoder_rnn_num_layers=2,
            encoder_rnn_dropout=0.1,
            speaker_embedding_dim=32,
        )
    ),
}
