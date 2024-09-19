import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

class FNetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        # Casting the inputs to complex64
        inp_complex = tf.cast(inputs, tf.complex64)
        # Projecting the inputs to the frequency domain using FFT2D and
        # extracting the real part of the output
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
class FNetDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
    

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


# Register the custom layers
tf.keras.utils.get_custom_objects().update({"FNetEncoder": FNetEncoder})
tf.keras.utils.get_custom_objects().update({"FNetDecoder": FNetDecoder})
tf.keras.utils.get_custom_objects().update({"PositionalEmbedding": PositionalEmbedding})

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Define the path to your model file
model_path = os.path.join(current_dir, 'models', 'chatbot_emotions.keras')

# Load the model
loaded_model = tf.keras.models.load_model(model_path)

# # Load the model
# loaded_model = tf.keras.models.load_model('models/chatbot_emotions.keras')

VOCAB_SIZE = 8192
MAX_LENGTH = 100
BATCH_SIZE = 64

def preprocess_text(sentence):
    sentence = tf.strings.lower(sentence)
    # Adding a space between the punctuation and the last word to allow better tokenization
    sentence = tf.strings.regex_replace(sentence, r"([?.!,])", r" \1 ")
    # Replacing multiple continuous spaces with a single space
    sentence = tf.strings.regex_replace(sentence, r"\s\s+", " ")
    # Replacing non-English words with spaces
    sentence = tf.strings.regex_replace(sentence, r"[^a-z?.!,]+", " ")
    # Removing "_comma_" from sentences
    sentence = tf.strings.regex_replace(sentence, "_comma_", ",")
    sentence = tf.strings.strip(sentence)
    sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
    return sentence

pkl_file_path = os.path.join(current_dir, 'models', 'qa.pkl')
with open(pkl_file_path, 'rb') as pkl_file:
    qa = pickle.load(pkl_file)


vectorizer = layers.TextVectorization(
    VOCAB_SIZE,
    standardize=preprocess_text,
    output_mode="int",
    output_sequence_length=MAX_LENGTH,
)

# We will adapt the vectorizer to both the questions and answers
# This dataset is batched to parallelize and speed up the process
vectorizer.adapt(tf.data.Dataset.from_tensor_slices((qa)).batch(BATCH_SIZE))

pkl_file_path = os.path.join(current_dir, 'models', 'vocab.pkl')
with open(pkl_file_path, 'rb') as pkl_file:
    uploaded_VOCAB = pickle.load(pkl_file)

def decode_sentence(input_sentence):
    # Mapping the input sentence to tokens and adding start and end tokens
    tokenized_input_sentence = vectorizer(
        tf.constant("[start] " + preprocess_text(input_sentence) + " [end]")
    )
    # Initializing the initial sentence consisting of only the start token.
    tokenized_target_sentence = tf.expand_dims(uploaded_VOCAB.index("[start]"), 0)
    decoded_sentence = ""

    for i in range(MAX_LENGTH):
        # Get the predictions
        predictions = loaded_model.predict(
            {
                "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0),
                "decoder_inputs": tf.expand_dims(
                    tf.pad(
                        tokenized_target_sentence,
                        [[0, MAX_LENGTH - tf.shape(tokenized_target_sentence)[0]]],
                    ),
                    0,
                ),
            }
        )
        # Calculating the token with maximum probability and getting the corresponding word
        sampled_token_index = tf.argmax(predictions[0, i, :])
        sampled_token = uploaded_VOCAB[sampled_token_index.numpy()]
        # If sampled token is the end token then stop generating and return the sentence
        if tf.equal(sampled_token_index, uploaded_VOCAB.index("[end]")):
            break
        decoded_sentence += sampled_token + " "
        tokenized_target_sentence = tf.concat(
            [tokenized_target_sentence, [sampled_token_index]], 0
        )
    decoded_sentence = decoded_sentence.replace('comma', ',') 
    
    return decoded_sentence
