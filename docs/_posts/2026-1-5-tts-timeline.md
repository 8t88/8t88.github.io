---
layout: post
title: TTS Timeline
permalink: /blog/tts_timeline
---

# A Timeline of TTS models

## A Decade of Innovation

January 5, 2026

Audio data has always captured my interest the most out of the various machine learning modalities. This is largely because of its seeming esoteric nature, particularly when it comes to speech vocalizations. If you were to look at a waveform plot or a spectrogram, it isn’t readily apparent how these data outputs map to words and phrases. With the explosive growth of generative machine learning in the past decade, I’ve tried to keep informed about the development of audio modeling, especially Text-to-Speech (TTS) systems.

The way I see it, audio has been the overlooked third child in the family of modalities, with its older siblings of text and image taking much of the spotlight. Technical challenges may have contributed to this: the data rate for audio is significantly higher than text, and there is a one-to-many relationship between text and audio, meaning that the same sentence can be rendered by different speakers with different speaking styles, emotional content, and recording conditions. Or maybe the lack of focus has just been because of people’s tastes. Whatever the reason, the capabilities of machine learning to generate audio recordings is only now reaching the level of astonishing realism we’ve gotten from text and image generation.

This post walks through the major Text-to-Speech models focused on or contributing to this development of audio generation, starting back in 2016\. There are plenty of forks and project-spinoffs out there, but these are openly published ([loosely speaking](https://arstechnica.com/information-technology/2024/08/debate-over-open-source-ai-term-brings-new-push-to-formalize-definition/)) models I’ve found to have gotten significant attention from their innovations or the resources devoted to them, and have led the way to our current state of the art. In later posts, I plan to compare/contrast the methodology and architectures these models use, and then deep dive into some specific models that I think are particularly interesting.

So let’s go back in time, starting in a year before we realized that Attention was all we needed…

## 2016

### [WaveNet](https://arxiv.org/pdf/1609.03499)

Starting things off in the pre-transformer days, Google’s WaveNet was the first major model to successfully use the raw waveform of the audio signal for input data, as opposed to feature engineering [acoustic features](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html). It uses an autoregressive architecture.  
This model was a significant advancement over the then state-of-the-art methods built with parametric and concatenative systems, and was able to generate English and Mandarin speech.

## 2017

### [Deep Voice](https://research.baidu.com/uploads/5ac03da1e126a.pdf)

Baidu iterated upon Deep Voice with some modifications and optimizations, first releasing Deep Voice 2 in March 2017\. This was followed by a complete architectural redesign resulting in [Deep Voice 3](https://arxiv.org/abs/1710.07654), released October 2017\. The redesign transformed the system from a combination of multiple neural networks into a single encoder-decoder design.

### [Tacotron](https://arxiv.org/pdf/1703.10135)

Another research development from Google, this system aimed to generate speech directly from text characters, rather than using the more prevalent at the time approach of multiple pipeline components including a linguistic feature extractor, vocoder, etc. The model took in \<text,audio\> pairs, transformed them into spectrograms, and trained on that data to produce waveforms.   
This sequence-to-sequence model uses an encoder-decoder architecture and an attention mechanism, potentially making it the first major text-to-speech transformer model.

## 2018

### [WaveGlow](https://arxiv.org/abs/1811.00002)

Submitted October 2018, this flow-based network model was built by Nvidia and used ideas from WaveNet and OpenAI’s [Glow](https://openai.com/index/glow/). Similar models include Glow-TTS and Flow-TTS. The system uses a diffusion model, taking mel-spectrograms as inputs, and is implemented with a single network for greater simplicity.   
One of the key design decisions of WaveGlow was to move away from an autoregressive system. While useful, autoregressive TTS systems are inherently sequential so have the downside of not being very parallelizable.

## 2019

### [wav2vec](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)

[Released](https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) by Meta in April 2019 and followed by wav2Vec 2.0 in 2020, this system uses raw, unlabeled audio to train automatic speech recognition models. It was trained by predicting speech units for masked parts of the audio, learning basic units that are 25ms long to enable learning of high-level contextualized representations

## 2020

### [EATS (End-to-End Adversarial Text-to-Speech)](https://arxiv.org/abs/2006.03575)

Released by DeepMind in June 2020, this system uses phonemes in an “end-to-end” manner, rather than having multiple processing stages (such as text normalisation, aligned linguistic featurization, mel-spectrogram synthesis, and raw audio waveform synthesis) The drawback of these multi-component systems is that they need extra training and tuning for each step of the pipeline, however the results appeared very promising.

## 2021

### [VITS](https://huggingface.co/docs/transformers/en/model_doc/vits) {#vits}

Released June 2021, this model also tried to [implement](https://github.com/jaywalnut310/vits/) a parallel end-to-end autoencoder system that matched the quality of two-stage systems. It used variational inference augmented with normalizing flows and an adversarial training process.  
This was followed by VITS2 in 2023, which [implemented](https://github.com/daniilrobnikov/vits2) some structure and training improvements 

### [HuBERT(Hidden-Unit BERT)](https://arxiv.org/abs/2106.07447)

While not technically a TTS system, this model is worth highlighting because of its use in other systems.  
Basically, this was Meta’s development of [BERT](https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/) for audio. Like, BERT, HuBERT, gets used in other audio models and processing tasks, such as [RVC](#heading=h.u0hhx0p1vuog)  
At a high level, HuBERT takes in a float array corresponding to the raw waveform of the speech signal and passes it to a wav2vec2.0 model to encode it. The final output is a feature sequence, and on that output the model uses a clustering step to create "hidden units".  
Read [here](https://jonathanbgn.com/2021/10/30/hubert-visually-explained.html) for a great in-depth examination of this system.

## 2022

2022 was the year that gen AI really hit the public consciousness, with the releases of Dall-E 2 and ChatGPT. While primary focus was on textual and visual results, this year still saw some very interesting experimentation and development in the audio space.

### [StyleTTS](https://styletts.github.io/)

The [StyleTTS paper](https://arxiv.org/abs/2205.15439) was initially submitted May 2022 and was followed-up by the [StyleTTS2](https://arxiv.org/abs/2306.07691) paper in June 2023 along with a code [implementation](https://styletts.github.io/). The goal was to develop a better parallel TTS system that improves on the deficiencies in other systems around aligning monotonic speech segments. The system uses an encoder-decoder architecture to generate a mel-spectrogram from text while incorporating parallel encoder blocks to be able to incorporate information around “style”, e.g. prosodic patterns and emotional tone. It also introduced an innovation called a Transferable Monotonic Aligner (TMA) to set the duration generation.   
This system has given rise to some interesting [forks](https://github.com/NeuralVox/StyleTTS2), [fine-tunes](https://github.com/IIEleven11/StyleTTS2FineTune?tab=readme-ov-file), and eventual [offshoots](#kokoro). 

### [AudioLM](https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/)

Released by Google in October 2022\. This model aims to generate realistic speech (and piano music) by listening to audio only.  
The system extracts semantic tokens using [w2v-BERT](https://arxiv.org/abs/2108.06209), as well as acoustic tokens produced by [a SoundStream neural codec](https://ai.googleblog.com/2021/08/soundstream-end-to-end-neural-audio.html), and passes these tokens through multiple transformer models. To get the final result, the SoundStream decoder converts the encoded tokens into a waveform. This was done for research purposes only, so although the paper described the details of the architecture, the weights and code were not released.

## 2023

With the sky opened and the belief in GenAI soaring, 2023 was a year when the amount of new TTS models really started to take off, with some very impressive and innovative systems appearing that still see a little bit of use today.

### [Piper](https://piper.ttstool.com/)

This is a TTS engine and a set of voice packs built by the Piper Project, from the [Open Home Foundation](https://www.openhomefoundation.org/). It is a small model that uses the [VITS](#vits) architecture. The initial release came in January 2023, and it continues to be open-sourced and developed.

### [ParlerTTS](https://github.com/huggingface/parler-tts/tree/main)

A reproduction of the work described in the paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com/), Parler-TTS is an auto-regressive transformer-based model, with the aim of being lightweight and producing natural-sounding speech. 

### [VALL-E](https://www.microsoft.com/en-us/research/project/vall-e-x/)

The next offering from Microsoft would appear in 2023 as [VALL-E](https://arxiv.org/pdf/2301.02111). Unlike other text-to-speech methods that typically synthesize speech by manipulating waveforms, VALL-E generates discrete audio codec codes from text and acoustic prompts. It basically analyzes how a person sounds, breaks that information into discrete components (called "tokens") using EnCodec, and uses training data to match what it "knows" about how that voice would sound if it spoke other phrases outside of the three-second sample.  
It was a pioneer in trying out language modeling for zero-shot TTS (TTS not trained for one specific speaker).

### [Bark](https://huggingface.co/suno/bark)

This transformer based model was open-sourced by [suno.ai](http://suno.ai) in April 2023\. The system consists of three components that form a pipeline: text-to-semantic tokens, semantic-to-coarse tokens, and coarse-to-fine tokens.  
This model was aimed toward producing more natural-sounding speech, as results generated contain non-verbal aspects such as laughing, as well as verbal tics of “umm”, “uh”, etc.  
While no longer maintained, this model can still be useful as a reference point or for research purposes. 

### [TorToiSe](https://github.com/neonbjb/tortoise-tts)

This model is where I think we begin to see the ice breaking around truly impressive speech generation. Rumored by many to be the basis of ElevenLabs’ technological suite, the TorToiSe model, named thus to underscore how *slow* it is, uses both an autoregressive encoder-decoder as well as a diffusion component to enhance its ability to model in the continuous domain and focus on reproducing human voices realistically.   
TorToiSe provides the autoregressive generator and DDPM with a "speech conditioning input", a unique design choice intended to allow the model to infer vocal characteristics like tone and prosody. In addition to the [paper](https://arxiv.org/pdf/2305.07243), you can read more about the design details [here](https://nonint.com/2022/04/25/tortoise-architectural-design-doc/).

### [AudioPaLM](https://google-research.github.io/seanet/audiopalm/examples/)

Released June 2023, this system combines two Google language models, PaLM-2 and AudioLM, into a multimodal architecture. It inherits the speaker identification and intonation copying capabilities from AudioLM, and the linguistic abilities of LLM from PaLM-2.

### [SeamlessM4T](https://about.fb.com/news/2023/08/seamlessm4t-ai-translation-model/)

Released by Meta in August 2023, this model performs text to speech as well as speech-to-speech and speech-to-text.  
An all-in-one model, it uses self-supervised speech representations with w2v-BERT 2.0, then uses the aligned the speech translations for training.

## 2024

Riding high on the momentum of 2023, this year saw even greater expansion and improvements. While companies such as OpenAI and Anthropic deployed closed systems with multimodal capabilities that could produce very impressive audio, open-source developments and modifications continued to emerge and push the space forward.

### [GPT-SoVITS-v2](https://rentry.co/GPT-SoVITS-guide#/)

Connected to the team involved with the [RVC voice changer](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and released in February 2024, this model generates acoustic tokens with a seq2seq model then converts those tokens into waveforms. Deep dives can be found [here](https://blog.openvino.ai/blog-posts/openvino-enable-digital-human-tts-gpt-sovits) and [here](https://medium.com/axinc-ai/gpt-sovits-a-zero-shot-speech-synthesis-model-with-customizable-fine-tuning-e4c72cd75d87). 

### [Metavoice](https://github.com/metavoiceio/metavoice-src)

Released in February 2024, this model predicts EnCodec tokens from text, and uses speaker information to condition the input at the token embedding layer. It then uses multi-band diffusion to generate waveforms from the EnCodec tokens, and concludes by passing the result through a DeepFilterNet to help clear up artifacts. The system also supports KV-caching via Flash Decoding and batching.

### [MeloTTS](https://huggingface.co/myshell-ai/MeloTTS-English)

A multi-lingual text-to-speech library by MIT and [MyShell.ai](http://MyShell.ai), this system was first released in February 2024\.   
The implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). The open-source codebase also [provides](https://github.com/myshell-ai/MeloTTS/blob/main/docs/training.md) some boilerplate to let you train and fine-tune the model on your own data.

### [E2-TTS](https://www.microsoft.com/en-us/research/project/e2-tts/) {#e2-tts}

Another model from Microsoft Research, this non-autoregressive system was released June 2024\. It uses only two modules, the flow-matching Transformer and the vocoder, eschewing additional components such as a duration model or grapheme-to-phoneme converter. It also does not use monotonic alignment or cross-attention.  
Despite detailing the architecture in the paper, this system was not open-sourced, but it was reverse-engineered by the [F5-TTS](#f5-tts) team as well as [some others](https://github.com/lucidrains/e2-tts-pytorch).

### [XTTS](https://arxiv.org/abs/2406.04904)

Coming out in June 2024, this system built upon the TorToiSe model, with some modifications.  
It aimed to expand language capabilities much wider to cover 16 languages, as well as provide cross-language zero-shot TTS without needing a parallel training dataset.  
The system takes in a  mel-spectrogram as input, and it makes use of a VQ-VAE, the GPT-2 encoder, as well as a HifiGAN model to compute the final audio signal  
A deeper dive into the system can be found [here](https://erogol.com/2023/11/11/xtts-v2-release-and-notes).

### [F5-TTS](https://swivid.github.io/F5-TTS/) {#f5-tts}

Released in October 2024, F5 is a non-autoregressive text-to-speech system based on flow matching with a Diffusion Transformer. It follows in the flow-matching footsteps of [E2-TTS](#e2-tts) but tries to make the design easier to follow and improves training time and performance. F5 also adds some additional innovations such as an inference-time Sway Sampling strategy.

## 2025

While the models of 2024 produced impressive results, 2025 may have been the year when the uncanny valley was crossed. The models being released could not only produce natural-sounding audio, but they could also convincingly transfer outputs across accents or even languages.  
This year also saw more companies popping up as competitors to ElevenLabs, either centered on audio generation or incorporating it into multimodal AI agents and general world models. Furthermore, models such as [Kokoro](#kokoro) and [Spark-TTS](#spark-tts) would be released that could produce fast and impressive results from extremely small models.

### [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) {#kokoro}

This model is built with StyleTTS2 as a backbone, though it’s not properly a distillation because it also removes parts of the original base model such as several utility models used during training in order to make it smaller. Although only the inference code and weights were released, this system got a lot of attention for its very small size while still producing impressive results.   
In addition to the trimming described above, another reason for its small model parameter count is that it doesn't use a dynamic style diffusion method and instead uses ‘voicepacks’ for voice style transfer. Additionally, Kokoro outsources tokenization and phonemization, so that the only thing the model sees is phonemes and punctuation, further reducing the size.

### [Llasa](https://llasatts.github.io/llasatts/) {#llasa}

Released around the beginning of February, the team behind Llasa had two main goals:

- Explore the scaling of train-time and inference-time compute for speech synthesis  
- Provide a simplified framework with single-layer vector quantizer (VQ) codec and a single Transformer architecture

The researchers argued that a simplified architecture, like the transformer for text LLMs, allows for faster scientific exploration and progress. The scaling of train-time compute (e.g., increasing model size or training data) also consistently improves the naturalness of and expressiveness of the results.  
The system first uses Llama 3B (could also use 1B or 8B) to predict "speech tokens from the [XCodec2](https://huggingface.co/HKUSTAudio/xcodec2) codebook, and then decodes these tokens into audio. 

### [Zonos](https://huggingface.co/Zyphra)

The Zonos models were first [released](https://www.zyphra.com/post/beta-release-of-zonos-v0-1) in February 2025 by Zyphra, a company whose mission is “to build human-aligned AI that helps individuals and organizations reach their fullest potential”.  
The company released two models, the [1.6B transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) and [1.6B hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) (SSM model, first open-source of its kind for TTS) under an Apache 2.0 license, with the goal of highly expressive and natural-sounding audio that attempts to capture emotional variation.   
The transformer model uses an autoregressive pipeline, and makes use of audio tokens from [DAC vocoder](https://github.com/descriptinc/descript-audio-codec). It also takes a speaker embedding as an input for voice cloning capabilities, as well as other inputs such as speaking rate, pitch, sample rate, audio quality, and speaker emotions, producing speech outputs that are 44KHz.  
While it produces high quality outputs, some weaknesses include: audio artifacts at the beginning and end of generations; mistakes in text alignment (skipping words, repeated words, etc); and high quality but slower and more expensive inference. The model attempts to mitigate these by implementing some techniques such as: a delay codebook pattern from [Parler TTS](https://huggingface.co/parler-tts), multi-token prediction, and embedding merging.

### [Sesame CSM](https://github.com/SesameAILabs/csm)

With Sesame we see another company moving into the audio-generation field aimed at achieving as natural an experience as possible, as if you were actually speaking to someone in the real world. Their goal is to achieve “voice presence”, and the key component aims are: emotional awareness, conversational dynamics, contextual awareness, and consistent personality.  
The Conversational Speech Model (CSM) was released February 2025\. The CSM has three sizes: Tiny (1B backbone, 100M decoder), Small (3B backbone, 250M decoder), and Medium (8B backbone, 300M decoder).  
Sesame’s CSM is a single-stage model, as opposed to a model using a two-step process of generating semantic tokens followed by acoustic tokens (commonly with [RVQ](https://arxiv.org/abs/2107.03312)). The downside of a two-step is that it is difficult to ensure the semantic tokens fully capture the prosody during training. Also, RVQ can have a high latency.  
The model incorporates two autoregressive transformers, both variants of the Llama architecture. This requires a high level of memory, so the system uses compute amortization to help alleviate the memory footprint.  
More detailed info can be found in the [release announcement](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

### [IndexTTS](https://index-tts.github.io/)

Released in Feb 2025 by the Index SpeechTeam, this system was based on XTTS and aimed at improvements over other popular TTS systems by utilizing a simpler training process, more controllable usage, and faster inference speed. This was followed by [IndexTTS2](https://index-tts.github.io/index-tts2.github.io/) later in the year, which introduced some innovative techniques addressing the issue of audio-visual synchronization, with a primary use case of video dubbing. While still using an autoregressive model, this system makes use of methods for speech duration control and gives independent control over timbre and emotion.

### [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) {#spark-tts}

Built using Qwen 2.5, this system is similar to [Llasa](#llasa) but much smaller and with great results. Instead of relying on separate models to generate acoustic features, it directly reconstructs audio from the code predicted by the LLM.

### [Orpheus](https://huggingface.co/collections/canopylabs/orpheus-tts-67d9ea3f6c05a941c06ad9d2) 

[Released](https://canopylabs.ai/model-releases) in March by Canopy Labs, these models come in four different sizes, ranging from 3 billion parameters to 150 million parameters. The architecture uses Llama-3b as the backbone but utilizes two nonstandard design decisions: 7 tokens per sample frame decoded as a single flattened sequence, and using a non-streaming (CNN-based) tokenizer. The [weights](https://huggingface.co/collections/canopylabs/orpheus-tts) and [code](https://github.com/canopyai/Orpheus-TTS) are all open-sourced under an Apache-2.0 license.

### [OuteTTS](https://github.com/edwko/OuteTTS)

Though some initial versions of OuteTTS started to get released at the end of 2024, this system went through a couple of iterations until OuteAI [released](https://outeai.com/blog/outeai-1.0-update) the first major version of their toolsuite utilizing these models in June 2025\. As of the end of 2025, the most advanced model is the [Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B), While details about any innovations are scarce, this gives yet another example of an organization attempting to build a software product around audio generation while open-sourcing their TTS models.

### [VibeVoice](https://microsoft.github.io/VibeVoice/)

An open-source development from Microsoft, this model was released to the public in August and comes in three sizes: 0.5B, 1.5B, and Large. It supports streaming text input as well as long-form generation. It uses an “interleaved, windowed design”, where the text chunks are encoded in parallel as they come in while the audio is generated from a diffuse-based model. 

### [Maya1](https://huggingface.co/maya-research/maya1)

For the final entry in our timeline, another model built with the Llama-3B backbone was released in October from Maya Research. This model uses a 3B-parameter decoder-only transformer (Llama-style) to predict SNAC neural codec tokens instead of raw waveforms. Besides the small size, the emphasis here is on creating emotionally expressive outputs of any type of voice described. The result is an impressive showing from a small research group and is one to keep an eye on in the future.

## Closing Thoughts

With this, we’ve made our way to the beginning of 2026\. I’m sure there are many models that I’ve missed, but I tried to capture important or particularly interesting ones that have shaped the conversation over the past decade.  
I hope you’ve enjoyed this high-level walkthrough of TTS history. Stay tuned for an upcoming post, where I’ll take a stab at digging into some of the major design decisions and tradeoffs that have shaped this history. Happy New Year everyone\!