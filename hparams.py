from types import SimpleNamespace

hparams = SimpleNamespace(
    audio_processing="deep-conv-tts",
    n_mel_channels=80,
    # Encoder parameters
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embedding_dim=512,
    # Decoder parameters
    n_frames_per_step=1,  # currently only 1 is supported
    decoder_rnn_dim=1024,
    prenet_dim=256,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_decoder_dropout=0.1,
    # Mel-post processing network parameters
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,
)
