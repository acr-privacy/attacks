import keras
import keras_nlp as nlp
import numpy as np
import tensorflow_addons as tfa


def create_model(
    n_bits=19,
    n_classes=10,
    seq_len=48,
    embedding_dim=128,
    embedding_bias=False,
    embedding_dropout=0.1,
    n_encoders=4,
    n_heads=4,
    mlp_factor=1,
    encoder_dropout=0.1,
    sd_proba=0.1,
    post_encoder_norm=False,
) -> keras.Model:
    """
    Builds a compact transformer inspired by CVT from
    https://github.com/SHI-Labs/Compact-Transformers
    """
    inputs = keras.layers.Input(shape=(seq_len, n_bits))
    masked_inputs = keras.layers.Masking(mask_value=0)(inputs)

    # learnable embeddings
    feature_embeddings = keras.layers.Dense(
        units=embedding_dim, use_bias=embedding_bias
    )(masked_inputs)
    position_embeddings = nlp.layers.PositionEmbedding(
        sequence_length=seq_len
    )(feature_embeddings)

    encoded = feature_embeddings + position_embeddings
    encoded = keras.layers.Dropout(embedding_dropout)(encoded)

    # encoder block
    keep_proba = 1 - np.linspace(0, sd_proba, n_encoders)
    for i in range(n_encoders):
        # Self attention layer.
        x = keras.layers.LayerNormalization(epsilon=1e-5)(encoded)
        x = keras.layers.MultiHeadAttention(
            n_heads, embedding_dim, dropout=encoder_dropout
        )(x, x)
        x = keras.layers.Dropout(encoder_dropout)(x)

        if keep_proba[i] < 1:
            encoded = tfa.layers.StochasticDepth(
                survival_probability=keep_proba[i]
            )([encoded, x])
        else:
            encoded += x

        # Feedforward layer
        x = keras.layers.LayerNormalization(epsilon=1e-5)(encoded)
        x = keras.layers.Dense(
            units=embedding_dim * mlp_factor, activation="gelu"
        )(x)
        x = keras.layers.Dense(units=embedding_dim)(x)
        x = keras.layers.Dropout(encoder_dropout)(x)

        if keep_proba[i] < 1:
            encoded = tfa.layers.StochasticDepth(
                survival_probability=keep_proba[i]
            )([encoded, x])
        else:
            encoded += x

    if post_encoder_norm:
        encoded = keras.layers.LayerNormalization(epsilon=1e-5)(encoded)

    # sequential pooling
    weights = keras.layers.Dense(units=1)(encoded)
    weights = keras.layers.Softmax(axis=1)(weights)
    weights = keras.layers.Permute((2, 1))(weights)

    weighted_representation = keras.layers.Dot(axes=(2, 1))([weights, encoded])
    weighted_representation = keras.layers.Reshape((embedding_dim,))(
        weighted_representation
    )

    # final classification
    classification = keras.layers.Dense(n_classes)(weighted_representation)

    return keras.Model(inputs=inputs, outputs=classification)
