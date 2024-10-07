import daca
from transformers import AutoConfig, AutoModel, EncodecFeatureExtractor

from parler_tts import DACAConfig, DACAModel
from transformers import AutoConfig, AutoModel
from transformers import EncodecFeatureExtractor

AutoConfig.register("daca", DACAConfig)
AutoModel.register(DACAConfig, DACAModel)

# Download a model
model_path = daca.utils.download(model_type="44khz")
model = daca.DACA.load(model_path)

hf_daca = DACAModel(DACAConfig())
hf_daca.model.load_state_dict(model.state_dict())

hf_daca.push_to_hub("parler-tts/dac_44khZ_8kbps")
EncodecFeatureExtractor(sampling_rate=44100).push_to_hub("parler-tts/dac_44khZ_8kbps")
