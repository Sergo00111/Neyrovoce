# Neyrovoce
Имитация голоса

```python
# Импортируем необходимые библиотеки
import torch
import numpy as np
import librosa
from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis

# Загружаем конфигурацию и модель
CONFIG_PATH = "tts_model.conf.json" # путь к файлу конфигурации
MODEL_PATH = "tts_model.pth.tar" # путь к файлу модели
config = load_config(CONFIG_PATH)
model = setup_model(len(phonemes) if config.use_phonemes else len(symbols), config)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")["model"])
model.eval()

# Создаем процессор аудио
ap = AudioProcessor(**config.audio)

# Определяем текст для синтеза
text = "Привет, я искусственный интеллект, который может говорить как человек"

# Преобразуем текст в последовательность фонем или символов
if config.use_phonemes:
    from TTS.tts.utils.text import text_to_sequence
    sequence = np.array(text_to_sequence(text, [config.text_cleaner], config.phoneme_language, config.phoneme_style))[None, :]
else:
    from TTS.tts.utils.text import text_to_sequence, sequence_to_text
    sequence = np.array(text_to_sequence(text, [config.text_cleaner]))[None, :]
    print(sequence_to_text(sequence))

# Синтезируем речь
alignments, _, mel_postnet_spec, _, _ = synthesis(model, sequence, config, use_cuda=False)

# Преобразуем спектрограмму в аудио
audio = ap.inv_mel_spectrogram(mel_postnet_spec.T)

# Сохраняем аудио в файл
librosa.output.write_wav("output.wav", audio, ap.sample_rate)
```
 
