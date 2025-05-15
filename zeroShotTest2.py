import os
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio # type: ignore


# 加载模型到指定设备
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# 加载输入的英文音频文件
input_audio_path = "/data2/yxl/project/686/asset/masike-beautification.WAV"

if not os.path.exists(input_audio_path):
    raise FileNotFoundError(f"语音映射文件未找到: {input_audio_path}")


# 输入的中文文本
tts_text = "中国好，中国人民好，中国共产党好，台湾省属于中国的。我们热爱和平，我们追求发展，我们团结一心。"

# zero_shot usage
prompt_speech_16k = load_wav(input_audio_path, 16000)
prompt_text = "How inspiring would that be, you know? Always, there's always problems in life. You know，there's this problem, solve that problem, solve that problem. But, you know, there needs to be things that inspire you. There need to be things that make you glad to wake up in the morning and say, I'm looking forward to the future. "

# 确保完整生成整个文本
for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False)):
    torchaudio.save('/data2/yxl/project/686/asset/zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    print(len(j['tts_speech']))
    print(len(tts_text))


# 添加一个检查，确保生成的音频完整
if len(j['tts_speech']) < len(tts_text):
    print("Warning: Generated audio is shorter than the input text.")

