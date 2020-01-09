from types import SimpleNamespace

hparams = SimpleNamespace(
    # Audio Parameters
    max_wav_value=32768.0,
    sampling_rate=16000,
    filter_length=1024,
    hop_length=212,  # defined it such that for a video frame it corresponds three audio samples
    win_length=1024,
    n_mel_channels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0,
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
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,
    # Mel-post processing network parameters
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,
)
