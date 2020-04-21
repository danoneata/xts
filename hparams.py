from types import SimpleNamespace


def get_hparams(**kwargs):
    hparams = dict(
        # Audio processing parameters
        audio_processing="deep-conv-tts",
        n_mel_channels=80,
        sampling_rate=16_000,
        # Conv 3d parameters
        conv3d_num_filters=64,
        conv3d_kernel_size=(3, 5, 5),
        # Encoder parameters
        # encoder_kernel_size=5,
        # encoder_n_convolutions=3,
        encoder_rnn_num_layers=1,
        encoder_rnn_dropout=0.0,
        encoder_embedding_dim=512,
        # Speaker information
        speaker_embedding_dim=None,
        embedding_normalize=False,
        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_decoder_dropout=0.1,
        drop_frame_rate=0.0,
        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    )
    hparams.update(kwargs)
    return SimpleNamespace(**hparams)


HPARAMS = {
    "sven": get_hparams(
        model_type="sven",
        conv3d_num_filters=64,
        conv3d_kernel_size=(3, 5, 5),
        encoder_rnn_num_layers=1,
        encoder_rnn_dropout=0.0,
    ),
    # baseline / B
    "magnus": get_hparams(
        model_type="sven",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
    ),
    # speaker identity / SI
    "magnus-multi-speaker": get_hparams(
        model_type="sven",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
    ),
    # speaker embedding / SE
    "bjorn": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        model_speaker_type="generic",
        use_fixed_embeddings=False,
    ),
    # baseline / B + drop frames
    "magnus-drop-frames": get_hparams(
        model_type="sven",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        drop_frame_rate=0.2,
    ),
    # speaker identity / SI + drop frames
    "magnus-multi-speaker-drop-frames": get_hparams(
        model_type="sven",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        drop_frame_rate=0.2,
        model_speaker_type="generic",
    ),
    # speaker identity / SI + drop frames + linear speaker
    "magnus-multi-speaker-drop-frames-linear-speaker": get_hparams(
        model_type="sven",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        drop_frame_rate=0.2,
        model_speaker_type="linear",
        model_speaker_path="output/models/grid_multi-speaker_speaker-classifier.pth",
    ),
    # speaker embedding / SE + drop frames
    "bjorn-drop-frames": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        drop_frame_rate=0.2,
        model_speaker_type="generic",
        use_fixed_embeddings=False,
    ),
    # speaker embedding / SE norm
    "bjorn-norm": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        model_speaker_type="generic",
        use_fixed_embeddings=False,
        embedding_normalize=True,
    ),
    # speaker embedding / SE norm + drop frames
    "bjorn-norm-drop-frames": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        drop_frame_rate=0.2,
        model_speaker_type="generic",
        use_fixed_embeddings=False,
        embedding_normalize=True,
    ),
    # speaker embedding / SE norm + drop frames + linear speaker
    "bjorn-norm-drop-frames-linear-speaker": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        drop_frame_rate=0.2,
        model_speaker_type="linear",
        use_fixed_embeddings=False,
        embedding_normalize=True,
    ),
    # speaker embedding / SE + drop frames + linear speaker
    "bjorn-drop-frames-linear-speaker": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        drop_frame_rate=0.2,
        model_speaker_type="linear",
        # TODO
        # model_speaker_path="output/models/grid_multi-speaker_speaker-classifier.pth",
        use_fixed_embeddings=False,
    ),
    "bjorn-fixed-emb": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        model_speaker_type="generic",
        use_fixed_embeddings=True,
    ),
    "bjorn-norm-fixed-emb": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        model_speaker_type="generic",
        use_fixed_embeddings=True,
    ),
    "bjorn-norm-fixed-emb-drop-frames-linear-speaker": get_hparams(
        model_type="bjorn",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
        drop_frame_rate=0.2,
        model_speaker_type="linear",
        use_fixed_embeddings=True,
    ),

    # speaker identity / SI
    "magnus-multi-speaker-2": get_hparams(
        model_type="sven-2",
        conv3d_num_filters=128,
        conv3d_kernel_size=(5, 5, 5),
        encoder_rnn_num_layers=2,
        encoder_rnn_dropout=0.1,
        speaker_embedding_dim=32,
    ),
}

for δ in [0.1, 0.2, 0.4]:
    for λ in [-5, -4, -3]:
        HPARAMS[f"bjorn-drop-frames-{δ}-linear-speaker-{λ}"] = get_hparams(
            model_type="bjorn",
            conv3d_num_filters=128,
            conv3d_kernel_size=(5, 5, 5),
            encoder_rnn_num_layers=2,
            encoder_rnn_dropout=0.1,
            speaker_embedding_dim=32,
            drop_frame_rate=δ,
            use_fixed_embeddings=False,
            model_speaker_type="linear",
            loss_speaker_weight=2 * 10 ** λ,
        )
