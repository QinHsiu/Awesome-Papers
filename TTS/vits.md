## TTS_vits
##### 记录一些关于在vits模型上进行改进的TTS模型，包括论文方法解读、代码资源等；

### 21年以及之前的文章
-《Efficient Neural Audio Synthesis》 ICML2018
  - 使用单层RNN搭配softmax来合成语音，其创新点主要有以下几点
  - 1.提出使用单层RNN的网络架构，使用了双层softmax，可以实时生成24k 16bit的语音；
  - 这里将原始输出看作两部分，高字节（细粒度的）和低字节（粗粒度的），给定前一时刻的粗粒度的表示$c_{t-1}$和细粒度表示$$f_{t-1}$$，使用公式得到下一个时刻的粗粒度的表示$$c_{t}$$，然后使用这三个数据作为输入生成下一个时刻的细粒度的表示$$c_{t}$$，这里可以使用的表达式如下：
  $$p(c_{t})=f_{c}(c_{t-1},f_{t-1}),p(f_{t})=f_{f}(c_{t-1},f_{t-1},c_{t})$$
  - 2.采用了权重剪枝的技术，实现了96%的稀疏性；
  随机初始化一个矩阵，每训练一定的steps就将矩阵中最小的k个元素置换为0.，k可以是动态的（递增的），也可以是预先设置好的；
  - 3.高并行度生成语音算法，把很长的序列生成折叠成若干个短的序列，每一个短序列同时生成，提高长序列的生成速度；
对于给定的128bit的数据，可以看作[1,2,...,128]，现在将其调整为[8*16]大小的矩阵，数据从下往上，从左至右依次排列，现在可以达到一个下降采样的策略，其首先生成的是最下面的数据[1,9,17,...]，在生成37、76、107的时候可以实现并行，最后将生成的数据变换为原始大小，也即实现了高度并行快速生成语音的目的；

  - 论文：https://arxiv.org/pdf/1802.08435.pdf
  - code：https://github.com/fatchord/WaveRNN
  - 学习资料：https://www.jianshu.com/p/b3019f2773ed

-《Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis》ICML2018
  - 1.引入一个reference encoder来学习具体的style embedding，在inference阶段直接使用训练好的reference encoder编码音频并将得出的style embedding加入到下游任务中，这里有几个问题：
    - style embedding的shape是不是预先定义的，也就是style的风格数目是不是一开始就确定了（这里如果事先确定可以不可以使用k-mens聚类来学习具体的style embedding）？
    - ps：这里的style embedding最后作为下游任务中attention的query融入到模型中，这里的style是从训练数据中提取出来的，其风格数目一开始就确定了；这里的reference encoder使用的是CNN+GRU，STL使用的是multi-head self-attention，一个GST=》一个reference encoder+一个STL模型（多头注意力模型）；
  $$Attention(Q,K,V)=softmax(\frac{Q\cdot K^{T}}{\sqrt{d/h}})\cdot V$$，transformer中encoder 和decoder连接部分是将ecoder的输出中的key和value传递到decoder中，然后与decoder中第一个block中的value进行上式计算获得；
    - 2.改进思路：
    - 0).将注意力得分哪里就是计算具体的style类型哪里是不是可以替换为无监督聚类，直接使用聚类中心表示具体的风格类型；
    - 1).是否可以引入对比学习，让学到的模型见到多种多样的embedding表示；

    - 论文：https://arxiv.org/pdf/1803.09017.pdf
    - code：https://github.com/KinglittleQ/GST-Tacotron
    - 学习资料：https://zhuanlan.zhihu.com/p/380487271
- 《One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization》InterSpeech 2019
  - 该论文目标是解决并行语料带来的两个问题：
  - （1）训练数据要求较为严格（并行数据）；
  - （2）只能转换处于训练集内的音色；
  - 本文通过instance normalization等技术进行音色和内容分离，然后重组音色和内容，达到生产目标音色的目的；关于音色转换的方法主要有两种（基于并行语料的有监督学习和基于非并行语料的无监督学习），其模型结构如上图所示，主要包含四个部分，三个encoder和一个AdaIN（自适应距离标准化），使用该模块将音色信息加入到音频内容信息上，其公式如下所示：
  $$\mu_{c}=\frac{1}{W}\sum^{W}_{\omega=1}M_{c}[\omega],\sigma_{c}=\sqrt{\frac{1}{W}\sum^{W}_{\omega=1}(M_{c}[\omega]-\mu_{c})^{2}+\epsilon}$$
  $$M'_{c}[\omega]=\frac{M_{c}[\omega]-\mu}{\sigma_{c}}$$,c表示第c个channel，W表示权重集合，w表示第w个权重，无仿射变换也即在归一化的时候不加gamma与beta，adaIN中加入了仿射变换（线性变换）
  $$M'_{c}[\omega]=\gamma_{c}\frac{M_{c}[\omega]-\mu}{\sigma_{c}}+\beta_{c}.$$
  - 其在encoder和decoder上都是使用的一维卷积实现，其中speak encoder和content encoder都使用了convBank，用于扩大感受野，instance normalization用于-   - content encoder，用来去除全局静态信息，这里核心模块AdaIN引入了speak Encoder的信息。
  - 论文：https://arxiv.org/pdf/2111.12277.pdf
  - code：https://github.com/jjery2243542/adaptive_voice_conversion
  - 学习资料：https://zhuanlan.zhihu.com/p/166098536
- 《LPCNET: IMPROVING NEURAL SPEECH SYNTHESIS THROUGH LINEAR PREDICTION 》ICASSP2019
  - 1.在WaveRNN的基础上引入LPC结构，该结构把source-filter部分的source使用神经网路来进行预测，而filter部分则使用DSP的方法来进行计算，其具体表达式如下所示：
  $$x_{n}=e_{n}+p_{n},p_{n}=\sum^{M}_{i}\alpha_{i}x_{n-i}$$
  - 2.其神经网络只对source的激励e进行预测，而filter的部分直接进行计算获取。该文章的作者回答这样的做的原因，是他认为使用一个神经网络不能同时很好的推算source和filter的两部分信息;
  - 论文：https://arxiv.org/pdf/1810.11846.pdf
  - code：https://github.com/xiph/LPCNet
  - 学习资料：https://zhuanlan.zhihu.com/p/321798376
- 《Improving LPCNet-based Text-to-Speech with Linear Prediction-structured Mixture Density Network》ICASSP2020
  - 1.提出基于LP-MDN的声码器ilpcnet，该声码器可以充分利用lp和激励之间的关系，替换原来的激励u-law离散分布，使合成音质提高;
  - 2.提出训练和生成的策略，比如stft loss等等。考虑到采样点和激励之间的分布关系，本文提出LP-MDN（mdn的均值、方差等参数是网络产生）;

  - 论文：https://arxiv.org/pdf/2001.11686.pdf
  - code: https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
  - 学习资料：https://zhuanlan.zhihu.com/p/321798376
- 《Bunched LPCNet : Vocoder for Low-cost Neural Text-To-Speech Systems》InterSpeech2020
  - 1.改进LPCNet采样策略，使得模型的速度提升了1.5倍；
  - 2.GRUa和GRUb部分都是多点共享，Dual-FC和softmax是每点独享。但每个激励之间并不是独立的，从第二个激励参数e开始，其的输入需要前一个e与gru输出进行拼接，该部分也是一个自回归的模式。而且gruA的输入不再输入一个点的信息，而是输入多个点的信息。该结构通过共享GRU部分，按理说该部分的推理速度正比于bunch sample的大小。

  - 论文：https://arxiv.org/pdf/2008.04574.pdf
  - code：code
  - 学习资料：https://zhuanlan.zhihu.com/p/321798376
- 《Gaussian Lpcnet for Multisample Speech Synthesis》ICASSP2020
  - 1.使用Gaussian 采样替代原来的softmax。原来的音频需要进行u-law转换，使用8bit进行采样点表示（造成音质损失），则softmax的维度则为256。本文使用gaussian直接对16bit进行采样，则采样部分由原来lpcnet的dualfc(256)+softmax(256)替换成fc1(128)+fc2(2)，文章提到不需要对音频进行加重处理。
  - 2.本文章进行采样时每一步推理两个采样点(两点之间互不影响）；

  - 论文：论文链接
  - 学习资料：https://zhuanlan.zhihu.com/p/321798376
- 《Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech 》ICML21
  - a.数据预处理部分会先根据频谱的长度进行分桶，这里避免了一个batch内长度差异较大带来的训练耗时长，效果差等影响，另外查找分桶的号使用了二分查找一定程度上加快了查找速度；
  - b.模型部分使用的基础编码器是transformer，其核心的需要训练的模块有三部分（生成器、判别器、时长预测器），这里跟着视频：https://www.bilibili.com/video/BV1VG411h75N/?spm_id_from=333.880.my_history.page.click，可以过一遍模型中的实现细节还有一些公式中的难点；
  - c.模型中还使用了动态规划去寻找最优的路径，使用了cpython来进行处理；

  - 论文：https://arxiv.org/pdf/2106.06103.pdf
  - code：https://github.com/jaywalnut310/vits
  - 学习资料：https://zhuanlan.zhihu.com/p/571040094
### 22年的文章
- 《ONE-SHOT VOICE CONVERSION FOR STYLE TRANSFER BASED ON SPEAKER ADAPTATION》ICASSP2022
  - 该论文指出先前对于音频合成建模主要问题🈶️两种方法：
  - （1）在训练过程中将语音分解为内容和说话人表征，在推断阶段将音频的内容表征和目标说话人的表征进行融合生成目标语音；该方法学习到的内容表征中包含有风格信息以及说话人相关信息，从而导致转换的音频不太稳定；
  - （2）将语音表征为内容、说话人、音高等，这种方法进一步提升了模型的解耦能力，但是转换语音和目标说话人语音的说话人相似度仍然有差距，另外也有较多工作关注于说话人自适应来应对few-shot甚至one-shot；
  - 本文的解决方法主要有以下几个点：
  - （1）采用说话人归一化去除瓶颈特征中说话人相关的信息，这样有助于提高最终说话人相似度；
  - （2）引入权重正则防止模型在训练过程中出现过拟合导致性能下降；
  - 论文：https://arxiv.org/pdf/2111.12277.pdf
  - 学习链接：https://zhuanlan.zhihu.com/p/617995266
- S《Self-supervised Context-aware Style Representation for Expressive Speech Synthesis》InterSpeech 2022

  - 该论文指出当前在语音中对style进行建模的方法主要有两种：无监督的joint learning和有监督的训练，其中无监督会面临文本信息泄漏进入style encoder中；另外需要大量的音频和文本信息；使用有监督信息会面临打标签时候存在主观臆断的问题（标注人员），并且一个简单的标签很难反应语音风格的本质；
本文的解决办法主要有以下几点：
  - （1）引入对比学习来预训练style embedding，来区分不同的文本的音色，替换文本中的情感词语，将具有相似情感的两个样本作为正样本，一个batch内的其他样本作为负样本;Loss如下所示：
    $$l_{cl}=-log\frac{exp(cos(h_{i},\tilde{h_{i}}))/\tau}{\sum^{N}_{k=1}\mathbf{1}_{k\neq i}exp(cos(h_{i},\tilde{h_{k}})/\tau)}$$
  - （2）引入一个深度聚类Loss，Loss如下所示：
    $$p_{ik}=\frac{q^{2}_{ik}/\sum_{i}q_{ik}}{\sum_{k'}(q^{2}_{ik'}/\sum_i q_{ik'})}, q_{ik}=\frac{(1+||h_{i}-\mu_{k}||^{2}_{2})^{-\frac{\alpha+1}{2}}}{\sum^{K}_{k'=1}(1+||h_{i}-\mu_{k'}||^{2}_{2}/\alpha)^{-\frac{\alpha+1}{2}}}$$
    $$l_{clu}=KL(P||Q)=\sum_{i}\sum_{k}p_{ik}\log\frac{p_{ik}}{q_{ik}}$$
  - code实现参看：https://github.com/amazon-science/sccl/blob/main/training.py
  - 这两个部分是参考的文章《Supporting Clustering with Contrastive Learning》https://arxiv.org/pdf/2103.12953.pdf
  - 论文：https://arxiv.org/pdf/2206.12559.pdf
  - Demo：https://wyh2000.github.io/InterSpeech2022/
- S《Styletts: A style-based generative model for natural and diverse text-to-speech synthesis》

  - 该论文主要分为两个阶段，在训练阶段，引入了四个encoder和一个adain残差板块，这里主要关注的style部分，他主要还是做的speak_to_speak的任务，这里style encoder从mel频谱中提取出风格表示，然后将其通过Adain残差板块重构mel频谱，这里的Loss如下所示：

  - 其在inference阶段训练decoer和duration predictor，其目前还是只能处理单源style的情况，其使用两阶段训练模型还会增加时间开销；
  - 论文：https://arxiv.org/pdf/2205.15439.pdf
  - 代码：https://github.com/yl4579/StyleTTS
- S《STYLETTS-VC: ONE-SHOT VOICE CONVERSION BY KNOWLEDGE TRANSFER FROM STYLE-BASED TTS MODELS》SLT2022

  - 上一篇论文的改进版，其主要用于单一样本的音频合成
  - 论文：https://arxiv.org/pdf/2212.14227.pdf
  - code：GitHub - yl4579/StyleTTS-VC: Official Implementation of StyleTTS-VC
  - demo：https://styletts-vc.github.io/
- 《FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis 》IJCAI2022

  - 加入Time-Aware LVC模块，用于提升音频合成速度
  - 论文：https://arxiv.org/pdf/2204.09934.pdf
  - code：https://github.com/Rongjiehuang/FastDiff
  - demo：https://fastdiff.github.io/
- 《HierSpeech: Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis 》 NeurIPS2022

  - 引入先验知识，使用预训练模型Wave2vec2.0来提取信息，并使用KL损失来进行约束，这里其他部分与vits一样
  - 论文：https://openreview.net/pdf?id=awdyRVnfQKX
  - 作者主页：https://github.com/sh-lee-prml
  - demo：https://sh-lee-prml.github.io/hierspeech-demo/
- 《Glow-WaveGAN 2: High-quality Zero-shot Text-to-speech Synthesis and Any-to-any Voice Conversion 》InterSpeech2022
  - 主要关注零样本音频合成
  - 论文：https://arxiv.org/pdf/2207.01832.pdf
  - 学习链接：https://zhuanlan.zhihu.com/p/547672526
  - demo：https://leiyi420.github.io/glow-wavegan2/
- 《A Multi-Stage Multi-Codebook VQ-VAE Approach to High-Performance Neural TTS 》InterSpeech 2022

  - 论文：https://arxiv.org/pdf/2209.10887.pdf
  - code：https://github.com/NVIDIA/NeMo
- S《VStyclone: Real-time Chinese voice style clone 》2022

  - 这篇文章提取风格表示的方法与GST一样，但是在提取的地方和合成的地方有差异；
  - 论文：论文链接
  - code：https://github.com/babysor/MockingBird
- S《GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech》NeurIPS 2022

  - 论文：论文链接 
  - PPT：https://neurips.cc/media/neurips-2022/Slides/54425.pdf
  - demo：https://generspeech.github.io/
- S《Disentangling Style and Speaker Attributes for TTS Style Transfer》ASP 2022

  - 该论文创新的地方在于为style单独设置了一个判别器用于专门学习该音频的style
  - 论文：https://arxiv.org/pdf/2201.09472.pdf
  - demo：https://xiaochunan.github.io/disentangling/index.html
- S《FINE-GRAINED STYLE CONTROL IN TRANSFORMER-BASED TEXT-TO-SPEECH SYNTHESIS》ICASSP 2022

  - 该论文使用了预训练模型Wave2Vec来提取音频风格和其他特征，通过GST和LST模块来抽取最终style表示；
  - 论文：https://arxiv.org/pdf/2110.06306.pdf
  - 代码：https://github.com/b04901014/FG-transformer-TTS
  - demo：https://b04901014.github.io/FG-transformer-TTS/
- S《CROSS-SPEAKER STYLE TRANSFER FOR TEXT-TO-SPEECH USING DATA AUGMENTATION》
  - 论文：https://arxiv.org/pdf/2202.05083.pdf
### 23年的文章
- S《IMPROVING THE QUALITY OF NEURAL TTS USING LONG-FORM CONTENT AND MULTI-SPEAKER MULTI-STYLE MODELING》ICASSP 2023
  - 论文：https://arxiv.org/pdf/2212.10075.pdf
- 《CONTEXTUAL EXPRESSIVE TEXT-TO-SPEECH》ICASSP2023
  
  - 论文：https://arxiv.org/pdf/2211.14548.pdf
  - demo：https://ofa-sys.github.io/Demo_CTTS/
- S《IMPROVING PROSODY FOR CROSS-SPEAKER STYLE TRANSFER BY SEMI-SUPERVISED STYLE EXTRACTOR AND HIERARCHICAL MODELING IN SPEECH SYNTHESIS》ICASSP 2023

  - 该论文使用半监督的方法来学习style，其使用了一部分标注style的数据混合未标注的数据一起训练模型
  - 论文：https://arxiv.org/pdf/2303.07711.pdf
  - Demo：https://qiangchunyu.github.io/style-transfer/STW.html
- 《DURATION-AWARE PAUSE INSERTION USING PRE-TRAINED LANGUAGE MODEL FOR MULTI-SPEAKER TEXT-TO-SPEECH》ICASSP 2023

  - 论文：https://arxiv.org/pdf/2302.13652.pdf
  - demo：https://ydqmkkx.github.io/pause-insertion/
- 《TEXT-TO-SPEECH SYNTHESIS BASED ON LATENT VARIABLE CONVERSION USING DIFFUSION PROBABILISTIC MODEL AND VARIATIONAL AUTOENCODER 》ICASSP2023
  - 论文：https://arxiv.org/pdf/2212.08329.pdf
- 《EVALUATING AND REDUCING THE DISTANCE BETWEEN SYNTHETIC AND REAL SPEECH DISTRIBUTIONS》ICASSP2023
  - 论文：https://arxiv.org/pdf/2211.16049.pdf
  - code：https://github.com/MiniXC/LightningFastSpeech2
- S《GRAD-STYLESPEECH: ANY-SPEAKER ADAPTIVE TEXT-TO-SPEECH SYNTHESIS WITH DIFFUSION MODELS 》ICASSP2023

  - 论文：https://arxiv.org/pdf/2211.09383.pdf
  - demo：https://nardien.github.io/grad-stylespeech-demo/
- 《PERIOD VITS: VARIATIONAL INFERENCE WITH EXPLICIT PITCH MODELING FOR END-TO-END EMOTIONAL SPEECH SYNTHESIS 》ICASSP2023

  - 论文：https://arxiv.org/pdf/2210.15964.pdf
  - demo：https://yshira116.github.io/period_vits_demo/
- 《LIGHTWEIGHT AND HIGH-FIDELITY END-TO-END TEXT-TO-SPEECH WITH MULTI-BAND GENERATION AND INVERSE SHORT-TIME FOURIER TRANSFORM 》ICASSP2023

  - 论文：https://arxiv.org/pdf/2210.15975.pdf
  - code：https://github.com/MasayaKawamura/MB-iSTFT-VITS
  - demo：https://masayakawamura.github.io/Demo_MB-iSTFT-VITS/
- 《CROSSSPEECH: SPEAKER-INDEPENDENT ACOUSTIC REPRESENTATION FOR CROSS-LINGUAL SPEECH SYNTHESIS 》ICASSP2023

  - 论文：https://arxiv.org/pdf/2302.14370.pdf
  - demo：https://lism13.github.io/demo/CrossSpeech/
- 《PROSODY-TTS: SELF-SUPERVISED PROSODY PRE- TRAINING WITH LATENT DIFFUSION FOR TEXT-TO- SPEECH 》NeurIPS2023 under view

  - 论文：openreview.net
- 《END-TO-END SPEECH SYNTHESIS BASED ON DEEP CONDITIONAL SCHRO ̈ DINGER BRIDGES 》NeurIPS2023 under view

  - 论文：https://openreview.net/pdf?id=K7YxdCYmd6w
  - demo：https://schron.github.io/
- 《EFFICIENTTTS 2: VARIATIONAL END-TO-END TEXT- TO-SPEECH SYNTHESIS AND VOICE CONVERSION 》NeurIPS2023 under view

  - 论文：https://openreview.net/pdf?id=__czv_gqDQt
- 《A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech 》 AAAI2023

  - 论文：https://arxiv.org/pdf/2302.04215.pdf
  - code：https://github.com/b04901014/MQTTS
- 《RWEN-TTS: Relation-aware Word Encoding Network for Natural Text-to-Speech Synthesis 》AAAI2023

  - 论文：https://arxiv.org/pdf/2212.07939.pdf
  - code：https://github.com/shinhyeokoh/rwen
- 《DINOISER: Diffused Conditional Sequence Learning by Manipulating Noises 》2023
  - 论文：https://arxiv.org/pdf/2302.10025.pdf
  - code：https://github.com/yegcjs/DINOISER
- 《FoundationTTS: Text-to-Speech for ASR Customization with Generative Language Model 》2023

  - 论文：https://arxiv.org/pdf/2303.02939.pdf
- 《LEVERAGING LARGE TEXT CORPORA FOR END-TO-END SPEECH SUMMARIZATION 》2023

  - 论文：https://arxiv.org/pdf/2303.00978.pdf
- 《UniFLG: Unified Facial Landmark Generator from Text or Speech 》2023

  - 论文：https://arxiv.org/pdf/2302.14337.pdf
  - demo：https://rinnakk.github.io/research/publications/UniFLG/
- 《PITS: Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS 》2023

  - 论文：https://arxiv.org/pdf/2302.12391.pdf
  - code：https://github.com/anonymous-pits/pits
- 《QuickVC: Any-To-Many Voice Conversion Using Inverse Short-Time Fourier Transform for Faster Conversion 》2023

  - 论文：https://arxiv.org/pdf/2302.08296.pdf
  - code：https://github.com/quickvc/QuickVC-VoiceConversion
  - demo：https://ericwudayi.github.io/VQVC-DEMO/
- *S《InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt 》2023

  - 论文：https://arxiv.org/pdf/2301.13662.pdf
  - demo：http://dongchaoyang.top/InstructTTS/
  - code：https://github.com/yangdongchao/InstructTTS
  - 参考code：https://github.com/yangdongchao/Text-to-sound-Synthesis
- 《Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers 》2023

  - 论文：https://arxiv.org/pdf/2301.02111.pdf
  - demo：https://valle-demo.github.io/
  - code：https://github.com/microsoft/unilm
- 《Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining 》2023

  - 论文：https://arxiv.org/pdf/2301.12596.pdf
  - demo：https://takaaki-saeki.github.io/zm-tts-text_demo/
- 《Controllable and Lossless Non-Autoregressive End-to-End Text-to-Speech 》2023

  - 论文：https://arxiv.org/pdf/2207.06088.pdf
  - demo：https://xcmyz.github.io/CLONE/

### 其他资源
- 端到端TTS模型: https://github.com/keonlee9420/Comprehensive-E2E-TTS
- TTSPaper：https://github.com/RevoSpeechTech/audio-generation-papers
- LM Audio：https://github.com/liusongxiang/Large-Audio-Models
- New TTS: https://github.com/wenet-e2e/speech-synthesis-paper
