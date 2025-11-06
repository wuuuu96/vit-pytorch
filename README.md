
## ViT的运行的动图

<img src="./images/vit.gif" width="500px"></img>

## Table of Contents

- [Vision Transformer - Pytorch](#vision-transformer---pytorch)
- [Install](#install)
- [Usage](#usage)
- [Parameters](#parameters)
- [Simple ViT](#simple-vit)
- [NaViT](#navit)
- [Distillation](#distillation)
- [Deep ViT](#deep-vit)
- [CaiT](#cait)
- [Token-to-Token ViT](#token-to-token-vit)
- [CCT](#cct)
- [Cross ViT](#cross-vit)
- [PiT](#pit)
- [LeViT](#levit)
- [CvT](#cvt)
- [Twins SVT](#twins-svt)
- [CrossFormer](#crossformer)
- [RegionViT](#regionvit)
- [ScalableViT](#scalablevit)
- [SepViT](#sepvit)
- [MaxViT](#maxvit)
- [NesT](#nest)
- [MobileViT](#mobilevit)
- [XCiT](#xcit)
- [Masked Autoencoder](#masked-autoencoder)
- [Simple Masked Image Modeling](#simple-masked-image-modeling)
- [Masked Patch Prediction](#masked-patch-prediction)
- [Masked Position Prediction](#masked-position-prediction)
- [Adaptive Token Sampling](#adaptive-token-sampling)
- [Patch Merger](#patch-merger)
- [Vision Transformer for Small Datasets](#vision-transformer-for-small-datasets)
- [3D Vit](#3d-vit)
- [ViVit](#vivit)
- [Parallel ViT](#parallel-vit)
- [Learnable Memory ViT](#learnable-memory-vit)
- [Dino](#dino)
- [EsViT](#esvit)
- [Accessing Attention](#accessing-attention)
- [Research Ideas](#research-ideas)
  * [Efficient Attention](#efficient-attention)
  * [Combining with other Transformer improvements](#combining-with-other-transformer-improvements)
- [FAQ](#faq)
- [Resources](#resources)
- [Citations](#citations)

## Vision Transformer - Pytorch

这是一个在 PyTorch环境中实现的 <a href="https://openreview.net/pdf?id=YicbFdNTTy">ViT模型</a>的库,其中ViT模型只使用单个 Transformer 编码器，就能在视觉分类任务中达到最先进（SOTA）的效果。 关于其重要性的进一步讲解，可以参考 <a href="https://www.youtube.com/watch?v=TrdevFK_am4">Yannic Kilcher's</a> 视频.其实代码量并不多，但我们把它整理出来，
是为了让更多人能够快速上手并加速这场“注意力机制革命”。
如果你想要使用的是**带预训练模型的 PyTorch 实现**，请参考 Ross Wightman 的仓库： <a href="https://github.com/rwightman/pytorch-image-models">here</a>.

Vision Transformer 的 **官方 JAX 实现** 在这里： <a href="https://github.com/google-research/vision_transformer">here</a>.

还有一个由研究科学家 Junho Kim 编写的 **TensorFlow 2** 版本：<a href="https://github.com/taki0112/vit-tensorflow">here</a>

此外，还有一个由 Enrico Shippole创建的 **Flax 实现版本**：<a href="https://github.com/conceptofmind/vit-flax">Flax translation</a> 

## Install(安装)

```bash
$ pip install vit-pytorch
```

## Usage(用法)

```python
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## Parameters(参数)

- `image_size`: 整型.  
图像尺寸。如果你的图像是矩形的，请确保该值为图像宽度和高度中的最大值。
- `patch_size`: int.  
patches尺寸。 `image_size` 必须能被 `patch_size`整除.  
Patch 的总数量为: ` n = (image_size // patch_size) ** 2` 并且 `n` 必须大于 16.
- `num_classes`: int.  
分类任务中的类别数量。
- `dim`: int.  
经过线性变换 `nn.Linear(..., dim)` 后输出张量的最后一个维度大小。
- `depth`: int.  
Transformer 模块（Block）的层数。
- `heads`: int.  
多头注意力层（Multi-head Attention）中的注意力头数。
- `mlp_dim`: int.  
MLP（前馈网络，FeedForward）层的维度大小。
- `channels`: int, default `3`.  
输入图像的通道数（例如 RGB 图像为 3 通道）。
- `dropout`: float between `[0, 1]`, default `0.`.  
Dropout 随机失活的比例。
- `emb_dropout`: float between `[0, 1]`, default `0`.  
Embedding 层的 Dropout 比例。
- `pool`: 表示池化方式，可以是`cls`表示使用分类 token（class token）进行池化；也可以是`mean`表示使用平均池化（mean pooling）。


## Simple ViT(简化版 ViT)

由原论文部分作者发布的<a href="https://arxiv.org/abs/2205.01580">An update</a> 提出了对 `ViT` 的多项简化改进，
这些改进使模型能够训练得更快、效果更好。

主要改进包括：

使用 二维正弦位置嵌入（2D sinusoidal positional embedding）；

使用 全局平均池化（Global Average Pooling），取消了原始的 CLS token；

去除 Dropout；

将批量大小从 4096 降至 1024；

采用 RandAugment 与 MixUp 数据增强方法；

研究发现，使用一个简单的线性层（Linear Head） 代替原始的 MLP 头部，性能并不会显著下降。

你可以按照下面的示例代码来导入并使用 SimpleViT：

```python
import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## NaViT

<img src="./images/navit.png" width="450px"></img>

<a href="https://arxiv.org/abs/2307.06304">This paper</a> 提出了一种方法，利用注意力机制（attention）和掩码机制（masking）在处理可变长度序列时的灵活性，
从而实现在同一批次（batch）中训练多分辨率图像。

这种方法在训练速度上显著提升，并且能够获得更高的精度，
其代价只是在模型结构与数据加载（dataloader）上引入了一定的额外复杂性。

论文中还采用了以下关键技术：

分解式二维位置编码（factorized 2D positional encodings）；

token dropping（令牌丢弃）；

query-key 归一化（query-key normalization）。

你可以按照下面的方式来使用该方法。

```python
import torch
from vit_pytorch.na_vit import NaViT

v = NaViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
)

# 5 images of different resolutions - List[List[Tensor]]

# for now, you'll have to correctly place images in same batch element as to not exceed maximum allowed sequence length for self-attention w/ masking

images = [
    [torch.randn(3, 256, 256), torch.randn(3, 128, 128)],
    [torch.randn(3, 128, 256), torch.randn(3, 256, 128)],
    [torch.randn(3, 64, 256)]
]

preds = v(images) # (5, 1000) - 5, because 5 images of different resolution above

```

或者，如果你希望框架自动将图像分组成可变长度的序列，
并确保这些序列的总长度不超过设定的最大长度（max length），
也可以使用这种方式。

```python
images = [
    torch.randn(3, 256, 256),
    torch.randn(3, 128, 128),
    torch.randn(3, 128, 256),
    torch.randn(3, 256, 128),
    torch.randn(3, 64, 256)
]

preds = v(
    images,
    group_images = True,
    group_max_seq_len = 64
) # (5, 1000)
```

最后，如果你想使用一种基于 <a href="https://pytorch.org/tutorials/prototype/nestedtensor.html">nested tensors（嵌套张量）</a> 的 NaViT 变体，
这种方式可以省去大量的掩码（masking）和填充（padding）操作，
请确保你的 PyTorch 版本为 2.5 或更高，
并按照以下方式导入使用。

```python
import torch
from vit_pytorch.na_vit_nested_tensor import NaViT

v = NaViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.,
    emb_dropout = 0.,
    token_dropout_prob = 0.1
)

# 5 images of different resolutions - List[Tensor]

images = [
    torch.randn(3, 256, 256), torch.randn(3, 128, 128),
    torch.randn(3, 128, 256), torch.randn(3, 256, 128),
    torch.randn(3, 64, 256)
]

preds = v(images)

assert preds.shape == (5, 1000)
```

## Distillation

<img src="./images/distill.png" width="300px"></img>

一篇最新的 <a href="https://arxiv.org/abs/2012.12877">paper</a> 表明，
在视觉 Transformer 中引入一种 “蒸馏标记（distillation token）”，
可以将卷积神经网络（CNN）中的知识**蒸馏（distill）**到 Transformer 中，
从而获得更小、更高效的视觉 Transformer 模型。

本仓库提供了便捷的方法，帮助你轻松实现这种蒸馏过程。

例如：
可以将 ResNet50（或其他教师模型） 的知识蒸馏到一个 Vision Transformer 中。

```python
import torch
from torchvision.models import resnet50

from vit_pytorch.distill import DistillableViT, DistillWrapper

teacher = resnet50(pretrained = True)

v = DistillableViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

distiller = DistillWrapper(
    student = v,
    teacher = teacher,
    temperature = 3,           # temperature of distillation
    alpha = 0.5,               # trade between main loss and distillation loss
    hard = False               # whether to use soft or hard distillation
)

img = torch.randn(2, 3, 256, 256)
labels = torch.randint(0, 1000, (2,))

loss = distiller(img, labels)
loss.backward()

# after lots of training above ...

pred = v(img) # (2, 1000)
```

DistillableViT 类与 ViT 几乎完全相同，
唯一的区别在于 前向传播（forward pass） 的处理方式。
因此，在完成蒸馏训练后，你可以将参数重新加载回普通的 ViT 模型中使用。

此外，你还可以直接调用 DistillableViT 实例自带的 .to_vit 方法，
快速将其转换回标准的 ViT 实例。

```python
v = v.to_vit()
type(v) # <class 'vit_pytorch.vit_pytorch.ViT'>
```


## Deep ViT

这篇 <a href="https://arxiv.org/abs/2103.11886">paper</a> 指出，
ViT 在较深层（超过 12 层）时难以有效捕捉注意力信息，
为此作者提出了一种解决方案 ——
在 softmax 之后对各个注意力头（attention head）的输出进行混合，
这种方法被称为 “再注意力（Re-attention）”。 

其实这一结果与自然语言处理领域中的 <a href="https://github.com/lucidrains/x-transformers#talking-heads-attention">Talking Heads注意力机制</a> 的研究结论非常一致。

你可以按照以下方式来使用该方法。

```python
import torch
from vit_pytorch.deepvit import DeepViT

v = DeepViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## CaiT

<a href="https://arxiv.org/abs/2103.17239">This paper</a> also notes difficulty in training vision transformers at greater depths and proposes two solutions. First it proposes to do per-channel multiplication of the output of the residual block. Second, it proposes to have the patches attend to one another, and only allow the CLS token to attend to the patches in the last few layers.

They also add <a href="https://github.com/lucidrains/x-transformers#talking-heads-attention">Talking Heads</a>, noting improvements

You can use this scheme as follows

```python
import torch
from vit_pytorch.cait import CaiT

v = CaiT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 12,             # depth of transformer for patch to patch attention only
    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05    # randomly dropout 5% of the layers
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## Token-to-Token ViT

<img src="./images/t2t.png" width="400px"></img>

这篇<a href="https://arxiv.org/abs/2101.11986">This paper</a> 提出， 在模型的前几层中可以通过 unfolding（展开） 操作对图像序列进行下采样，
从而使得每个 token 中包含部分重叠的图像信息（如上图所示）。

你可以按如下方式使用这种 `ViT` 的变体版本。

```python
import torch
from vit_pytorch.t2t import T2TViT

v = T2TViT(
    dim = 512,
    image_size = 224,
    depth = 5,
    heads = 8,
    mlp_dim = 512,
    num_classes = 1000,
    t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
)

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)
```

## CCT

<img src="https://raw.githubusercontent.com/SHI-Labs/Compact-Transformers/main/images/model_sym.png" width="400px"></img>

<a href="https://arxiv.org/abs/2104.05704">CCT</a> 提出了一种紧凑型 Transformer，
它通过卷积（convolution）替代传统的图像切块（patching）方式，并使用序列池化（sequence pooling）。

这种设计使得 CCT 同时具备较高的准确率和较少的参数量。

你可以通过以下两种方式来使用该模型。
```python
import torch
from vit_pytorch.cct import CCT

cct = CCT(
    img_size = (224, 448),
    embedding_dim = 384,
    n_conv_layers = 2,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 14,
    num_heads = 6,
    mlp_ratio = 3.,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)

img = torch.randn(1, 3, 224, 448)
pred = cct(img) # (1, 1000)
```

或者，你也可以直接使用几个预定义的模型`（[2, 4, 6, 7, 8, 14, 16]）`，
这些模型已经预先设定了以下参数：

层数（number of layers），

注意力头数（number of attention heads），

MLP 比例（mlp ratio），

嵌入维度（embedding dimension）。

```python
import torch
from vit_pytorch.cct import cct_14

cct = cct_14(
    img_size = 224,
    n_conv_layers = 1,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)
```

<a href="https://github.com/SHI-Labs/Compact-Transformers">Official
Repository官方仓库</a> 中提供了**预训练模型的权重（checkpoints）**下载链接。

## Cross ViT

<img src="./images/cross_vit.png" width="400px"></img>

<a href="https://arxiv.org/abs/2103.14899">This paper</a> 提出了一种结构，
使用 两个视觉 Transformer 在**不同的图像尺度（scales）上并行处理图像，
并且这两个 Transformer 会周期性地进行交叉注意（cross-attention）**以交换信息。

实验结果表明，这种方法在基础版 Vision Transformer 的性能之上取得了进一步的提升。
```python
import torch
from vit_pytorch.cross_vit import CrossViT

v = CrossViT(
    image_size = 256,
    num_classes = 1000,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
    lg_dim = 384,            # low res dimension
    lg_patch_size = 64,      # low res patch size
    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

pred = v(img) # (1, 1000)
```

## PiT

<img src="./images/pit.png" width="400px"></img>

<a href="https://arxiv.org/abs/2103.16302">This paper</a> 提出了一种方法，
通过使用**深度可分离卷积（depth-wise convolution）进行池化（pooling）**操作，
以实现对 token 的下采样（downsampling）。

```python
import torch
from vit_pytorch.pit import PiT

v = PiT(
    image_size = 224,
    patch_size = 14,
    dim = 256,
    num_classes = 1000,
    depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)
```

## LeViT

<img src="./images/levit.png" width="300px"></img>

<a href="https://arxiv.org/abs/2104.01136">This paper</a> 提出了多项改进，包括：

1)使用卷积嵌入（convolutional embedding）替代原先的分块投影（patch-wise projection）；

2)采用分阶段下采样（downsampling in stages）；

3)在注意力机制中引入额外的非线性（extra non-linearity in attention）；

4)使用二维相对位置偏置（2D relative positional biases），取代原始的绝对位置偏置（absolute positional bias）；

5)用 BatchNorm 替代 LayerNorm。

<a href="https://github.com/facebookresearch/LeViT">Official repository官方仓库</a>

```python
import torch
from vit_pytorch.levit import LeViT

levit = LeViT(
    image_size = 224,
    num_classes = 1000,
    stages = 3,             # number of stages
    dim = (256, 384, 512),  # dimensions at each stage
    depth = 4,              # transformer of depth 4 at each stage
    heads = (4, 6, 8),      # heads at each stage
    mlp_mult = 2,
    dropout = 0.1
)

img = torch.randn(1, 3, 224, 224)

levit(img) # (1, 1000)
```

## CvT

<img src="./images/cvt.png" width="400px"></img>

<a href="https://arxiv.org/abs/2103.15808">This paper</a> 提出了卷积与注意力机制相结合的方法。
具体来说：

使用卷积（convolutions）在三个阶段中对图像或特征图进行嵌入（embedding）和下采样（downsampling）；

同时使用**深度可分离卷积（depthwise convolution）**来生成注意力机制中的 query、key 和 value 投影。

```python
import torch
from vit_pytorch.cvt import CvT

v = CvT(
    num_classes = 1000,
    s1_emb_dim = 64,        # stage 1 - dimension
    s1_emb_kernel = 7,      # stage 1 - conv kernel
    s1_emb_stride = 4,      # stage 1 - conv stride
    s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
    s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
    s1_heads = 1,           # stage 1 - heads
    s1_depth = 1,           # stage 1 - depth
    s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
    s2_emb_dim = 192,       # stage 2 - (same as above)
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,       # stage 3 - (same as above)
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 4,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.
)

img = torch.randn(1, 3, 224, 224)

pred = v(img) # (1, 1000)
```

## Twins SVT

<img src="./images/twins_svt.png" width="400px"></img>

This <a href="https://arxiv.org/abs/2104.13840">paper</a> 提出了将局部注意力（local attention）与全局注意力（global attention）相结合的方法，
并引入了位置编码生成器（Position Encoding Generator, PEG） (最早由 <a href="https://arxiv.org/abs/2102.10882">CPVT提出</a>) 以及全局平均池化（Global Average Pooling）, 该方法在不使用 滑动窗口（shifted windows）、CLS token 或 位置嵌入（positional embeddings） 的情况下，达到了与 <a href="https://arxiv.org/abs/2103.14030">Swin</a>相当的效果，同时大大简化了模型结构。
```python
import torch
from vit_pytorch.twins_svt import TwinsSVT

model = TwinsSVT(
    num_classes = 1000,       # number of output classes
    s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
    s1_patch_size = 4,        # stage 1 - patch size for patch embedding
    s1_local_patch_size = 7,  # stage 1 - patch size for local attention
    s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
    s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
    s2_emb_dim = 128,         # stage 2 (same as above)
    s2_patch_size = 2,
    s2_local_patch_size = 7,
    s2_global_k = 7,
    s2_depth = 1,
    s3_emb_dim = 256,         # stage 3 (same as above)
    s3_patch_size = 2,
    s3_local_patch_size = 7,
    s3_global_k = 7,
    s3_depth = 5,
    s4_emb_dim = 512,         # stage 4 (same as above)
    s4_patch_size = 2,
    s4_local_patch_size = 7,
    s4_global_k = 7,
    s4_depth = 4,
    peg_kernel_size = 3,      # positional encoding generator kernel size
    dropout = 0.              # dropout
)

img = torch.randn(1, 3, 224, 224)

pred = model(img) # (1, 1000)
```

## RegionViT

<img src="./images/regionvit.png" width="400px"></img>

<img src="./images/regionvit2.png" width="400px"></img>

<a href="https://arxiv.org/abs/2106.02689">This paper</a> 提出了将**特征图（feature map）划分为多个局部区域（local regions）**的方法，
在每个局部区域内，局部 token 之间相互进行注意力计算（local tokens attend to each other）。

此外，每个局部区域还包含一个区域 token（regional token），
它不仅与该区域内的所有局部 token 进行交互，还会与其他区域的区域 token进行注意力计算。

这使得模型能够同时捕获局部细节信息与全局上下文关系。

可以按下面方法运行

```python
import torch
from vit_pytorch.regionvit import RegionViT

model = RegionViT(
    dim = (64, 128, 256, 512),      # tuple of size 4, indicating dimension at each stage
    depth = (2, 2, 8, 2),           # depth of the region to local transformer at each stage
    window_size = 7,                # window size, which should be either 7 or 14
    num_classes = 1000,             # number of output classes
    tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
    use_peg = False,                # whether to use positional generating module. they used this for object detection for a boost in performance
)

img = torch.randn(1, 3, 224, 224)

pred = model(img) # (1, 1000)
```

## CrossFormer

<img src="./images/crossformer.png" width="400px"></img>

<img src="./images/crossformer2.png" width="400px"></img>

This <a href="https://arxiv.org/abs/2108.00154">paper</a> 通过交替使用局部注意力（local attention）和全局注意力（global attention），在性能上超越了 PVT 和 Swin Transformer。

其中，全局注意力是在**窗口维度（windowing dimension）**上进行的，以此降低计算复杂度，这种机制与 轴向注意力（axial attention） 的设计思路类似。

此外，论文还提出了一个跨尺度嵌入层（cross-scale embedding layer），作者证明该层是一种通用模块，可用于提升各种视觉 Transformer 的性能。

同时，研究者还引入了动态相对位置偏置（dynamic relative positional bias），
使得模型能够泛化到更高分辨率的图像，进一步增强了模型的适应性与可扩展性。

```python
import torch
from vit_pytorch.crossformer import CrossFormer

model = CrossFormer(
    num_classes = 1000,                # number of output classes
    dim = (64, 128, 256, 512),         # dimension at each stage
    depth = (2, 2, 8, 2),              # depth of transformer at each stage
    global_window_size = (8, 4, 2, 1), # global window sizes at each stage
    local_window_size = 7,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
)

img = torch.randn(1, 3, 224, 224)

pred = model(img) # (1, 1000)
```

## ScalableViT

<img src="./images/scalable-vit-1.png" width="400px"></img>

<img src="./images/scalable-vit-2.png" width="400px"></img>

这篇由 字节跳动 AI 团队发表的 <a href="https://arxiv.org/abs/2203.10790">paper</a> 提出了两个新模块：
可扩展自注意力（Scalable Self Attention, SSA） 和 交互式窗口自注意力（Interactive Windowed Self Attention, IWSA）。

SSA 模块：通过将 Key / Value 特征图 按一定比例（reduction_factor）下采样来降低早期阶段的计算量，同时调整 Query 和 Key 的维度大小（ssa_dim_key），从而在保持性能的前提下减少计算开销。

IWSA 模块：在局部窗口中执行自注意力（类似于其他视觉 Transformer 方法），但额外加入了一个 局部交互模块（Local Interactive Module, LIM）。
LIM 将 Value 向量通过一个卷积核大小为 3 的卷积层后再以残差形式加入，从而增强局部特征的建模能力。

论文中声称这种架构的性能超过了 Swin Transformer，并在实验中展示了对 Crossformer 的竞争性结果。

例如，下面展示了如何使用 ScalableViT-S（小型可扩展视觉 Transformer）模型的示例用法。

```python
import torch
from vit_pytorch.scalable_vit import ScalableViT

model = ScalableViT(
    num_classes = 1000,
    dim = 64,                               # starting model dimension. at every stage, dimension is doubled
    heads = (2, 4, 8, 16),                  # number of attention heads at each stage
    depth = (2, 2, 20, 2),                  # number of transformer blocks at each stage
    ssa_dim_key = (40, 40, 40, 32),         # the dimension of the attention keys (and queries) for SSA. in the paper, they represented this as a scale factor on the base dimension per key (ssa_dim_key / dim_key)
    reduction_factor = (8, 4, 2, 1),        # downsampling of the key / values in SSA. in the paper, this was represented as (reduction_factor ** -2)
    window_size = (64, 32, None, None),     # window size of the IWSA at each stage. None means no windowing needed
    dropout = 0.1,                          # attention and feedforward dropout
)

img = torch.randn(1, 3, 256, 256)

preds = model(img) # (1, 1000)
```

## SepViT

<img src="./images/sep-vit.png" width="400px"></img>

另一篇由 字节跳动 AI 团队发表的 <a href="https://arxiv.org/abs/2203.15380">Bytedance AI paper</a>, 提出了一个 深度可分离-逐点自注意力层（Depthwise-Pointwise Self-Attention Layer），其设计灵感主要来源于 MobileNet 的深度可分离卷积结构。

其中最有趣的创新在于：将深度自注意力阶段产生的特征图（feature map）直接复用为逐点自注意力（pointwise self-attention）阶段的 Value 输入，如上图所示。这种跨阶段信息复用大大提升了特征利用率与效率。

我只在实现中保留了带有这种特定自注意力层的版本（称为 SepViT），
因为论文中提到的分组注意力（grouped attention）部分既不新颖，也未清楚说明其对窗口内 token 的处理方式。
此外，作者的实验表明，仅使用这个 DSSA 层（Depthwise Separable Self-Attention），
就已经能在性能上超越 Swin Transformer。

例如，可以使用轻量化版本 SepViT-Lite 来实现该结构。

```python
import torch
from vit_pytorch.sep_vit import SepViT

v = SepViT(
    num_classes = 1000,
    dim = 32,               # dimensions of first stage, which doubles every stage (32, 64, 128, 256) for SepViT-Lite
    dim_head = 32,          # attention head dimension
    heads = (1, 2, 4, 8),   # number of heads per stage
    depth = (1, 2, 6, 2),   # number of transformer blocks per stage
    window_size = 7,        # window size of DSS Attention block
    dropout = 0.1           # dropout
)

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)
```

## MaxViT

<img src="./images/max-vit.png" width="400px"></img>

<a href="https://arxiv.org/abs/2204.01697">This paper</a> 提出了一个卷积与注意力结合的混合网络结构，
在卷积部分使用了 MBConv 模块（来自 MobileNetV2 的高效卷积单元），
而在注意力部分采用了 块状（block）/ 网格（grid）轴向稀疏注意力机制（axial sparse attention）。

研究者还指出，这种特定结构的视觉 Transformer 在生成模型（例如 GAN）任务中表现良好，
能够兼顾特征提取能力与生成效果。

例如，可以使用轻量化版本 MaxViT-S 来实现该模型。
```python
import torch
from vit_pytorch.max_vit import MaxViT

v = MaxViT(
    num_classes = 1000,
    dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
    dim = 96,                         # dimension of first layer, doubles every layer
    dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
    depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
    window_size = 7,                  # window size for block and grids
    mbconv_expansion_rate = 4,        # expansion rate of MBConv
    mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
    dropout = 0.1                     # dropout
)

img = torch.randn(2, 3, 224, 224)

preds = v(img) # (2, 1000)
```

## NesT

<img src="./images/nest.png" width="400px"></img>

 <a href="https://arxiv.org/abs/2105.12723">paper</a> 提出了一种分层（hierarchical）视觉 Transformer 架构，
在不同的层次阶段对图像进行处理。

在每个阶段中，注意力（attention）仅在局部块（local block）内的 token 之间计算，
并且随着层级上升，这些局部块会逐步聚合（aggregation）成更高层次的表示。

这种聚合操作在**图像平面（image plane）上完成，
并且通过卷积（convolution）与后续的最大池化（maxpool）**来实现跨块（block boundary）的信息传递，
从而让模型既保留局部特征，又能整合全局上下文信息。

下面是使用该结构的示例代码，例如轻量化版本 NesT-T。

```python
import torch
from vit_pytorch.nest import NesT

nest = NesT(
    image_size = 224,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (2, 2, 8),  # the number of transformer blocks at each hierarchy, starting from the bottom
    num_classes = 1000
)

img = torch.randn(1, 3, 224, 224)

pred = nest(img) # (1, 1000)
```

## MobileViT

<img src="./images/mbvit.png" width="400px"></img>

 <a href="https://arxiv.org/abs/2110.02178">paper</a>介绍了 MobileViT，
一种为移动设备设计的轻量级通用视觉 Transformer。

MobileViT 提供了一种全新的思路，
它通过 Transformer 的全局信息处理能力，
在保持轻量化结构的同时，实现了对图像全局特征的有效建模。

你可以使用如下代码（例如 mobilevit_xs 版本）来调用该模型。

```python
import torch
from vit_pytorch.mobile_vit import MobileViT

mbvit_xs = MobileViT(
    image_size = (256, 256),
    dims = [96, 120, 144],
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    num_classes = 1000
)

img = torch.randn(1, 3, 256, 256)

pred = mbvit_xs(img) # (1, 1000)
```

## XCiT

<img src="./images/xcit.png" width="400px"></img>

 <a href="https://arxiv.org/abs/2106.09681">paper</a> 提出了交叉协方差注意力机制（Cross Covariance Attention, XCA）。

可以将它理解为：
与传统的注意力机制在**空间维度（spatial dimension）上计算注意力不同，
XCA 在特征维度（feature dimension）**上执行注意力计算。

换一种角度理解，
它相当于一种动态的 1×1 卷积（dynamic 1x1 convolution），
其中卷积核由**空间相关性（spatial correlations）**定义的注意力图（attention map）动态生成。

从技术上讲，
XCA 仅仅是在执行余弦相似度注意力（cosine similarity attention）之前，
对查询（query）、键（key）、值（value）进行转置（transpose），
并使用一个**可学习的温度参数（learned temperature）**来调节注意力分布。

```python
import torch
from vit_pytorch.xcit import XCiT

v = XCiT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 12,                     # depth of xcit transformer
    cls_depth = 2,                  # depth of cross attention of CLS tokens to patch, attention pool at end
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05,           # randomly dropout 5% of the layers
    local_patch_kernel_size = 3     # kernel size of the local patch interaction module (depthwise convs)
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## Simple Masked Image Modeling

<img src="./images/simmim.png" width="400px"/>

<a href="https://arxiv.org/abs/2111.09886">paper</a> 提出了一种**简单的掩码图像建模（SimMIM, Simple Masked Image Modeling）**方法。

在这种方法中，模型仅使用一个线性投影层（linear projection），
将被掩盖的图像 token 映射回像素空间，
然后与原始图像中被掩盖区域的像素值计算 L1 损失（L1 loss）。

尽管方法非常简单，但其实验结果与许多更复杂的掩码建模方法相比依然具有竞争力（competitive performance）。

你可以如下方式使用该方法。

```python
import torch
from vit_pytorch import ViT
from vit_pytorch.simmim import SimMIM

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mim = SimMIM(
    encoder = v,
    masking_ratio = 0.5  # they found 50% to yield the best results
)

images = torch.randn(8, 3, 256, 256)

loss = mim(images)
loss.backward()

# that's all!
# do the above in a for loop many times with a lot of images and your vision transformer will learn

torch.save(v.state_dict(), './trained-vit.pt')
```


## Masked Autoencoder

<img src="./images/mae.png" width="400px"/>

这篇由 何恺明（Kaiming He） 等人发表的 <a href="https://arxiv.org/abs/2111.06377">Kaiming He paper</a> 提出了一种简单的自编码器（autoencoder）结构。

在这种方法中，
视觉 Transformer（ViT）只对未被掩盖的图像块（unmasked patches）进行注意力计算，
而一个更小的解码器（decoder）负责重建被掩盖的像素值（masked pixel values）。

相关讲解视频：
<a href="https://www.youtube.com/watch?v=LKixq2S2Pz8">DeepReader quick paper review</a>

<a href="https://www.youtube.com/watch?v=Dp6iICL2dVI">AI Coffeebreak with Letitia</a>

你可以使用以下代码来实现该模型。

```python
import torch
from vit_pytorch import ViT, MAE

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mae = MAE(
    encoder = v,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

images = torch.randn(8, 3, 256, 256)

loss = mae(images)
loss.backward()

# that's all!
# do the above in a for loop many times with a lot of images and your vision transformer will learn

# save your improved vision transformer
torch.save(v.state_dict(), './trained-vit.pt')
```

## Masked Patch Prediction

 <a href="https://github.com/zankner">Zach</a>提出的**原始掩码图像块预测任务（masked patch prediction task）**

```python
import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,          # probability of using token in masked prediction task
    random_patch_prob=0.30,  # probability of randomly replacing a token being used for mpp
    replace_prob=0.50,       # probability of replacing a token being used for mpp with the mask token
)

opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)
    opt.zero_grad()
    loss.backward()
    opt.step()

# save your improved network
torch.save(model.state_dict(), './pretrained-net.pt')
```

## Masked Position Prediction

<img src="./images/mp3.png" width="400px"></img>

New <a href="https://arxiv.org/abs/2207.07611">paper</a> 提出了掩码位置预测（masked position prediction）的预训练策略。
这种方法相比于传统的掩码自编码器（Masked Autoencoder, MAE）策略具有更高的效率，
同时在性能上也能达到相当的水平（comparable performance）。

```python
import torch
from vit_pytorch.mp3 import ViT, MP3

v = ViT(
    num_classes = 1000,
    image_size = 256,
    patch_size = 8,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
)

mp3 = MP3(
    vit = v,
    masking_ratio = 0.75
)

images = torch.randn(8, 3, 256, 256)

loss = mp3(images)
loss.backward()

# that's all!
# do the above in a for loop many times with a lot of images and your vision transformer will learn

# save your improved vision transformer
torch.save(v.state_dict(), './trained-vit.pt')
```

## Adaptive Token Sampling

<img src="./images/ats.png" width="400px"></img>

<a href="https://arxiv.org/abs/2111.15667">paper</a> 提出了一种方法：
利用 CLS（分类）token 的注意力分数（attention scores），
并根据 value head 的范数（norms） 对这些分数进行重新加权，
从而在不同层中筛除（丢弃）不重要的 token，
以减少计算量并提升模型效率。
```python
import torch
from vit_pytorch.ats_vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    max_tokens_per_depth = (256, 128, 64, 32, 16, 8), # a tuple that denotes the maximum number of tokens that any given layer should have. if the layer has greater than this amount, it will undergo adaptive token sampling
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(4, 3, 256, 256)

preds = v(img) # (4, 1000)

# you can also get a list of the final sampled patch ids
# a value of -1 denotes padding

preds, token_ids = v(img, return_sampled_token_ids = True) # (4, 1000), (4, <=8)
```

## Patch Merger


<img src="./images/patch_merger.png" width="400px"></img>

<a href="https://arxiv.org/abs/2202.12015">paper</a> 提出了一个简单的模块——Patch Merger（块合并器），
它可以在视觉 Transformer 的任意层中减少 token（图像块表示）的数量，
而不会损失模型性能。

```python
import torch
from vit_pytorch.vit_with_patch_merger import ViT

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 12,
    heads = 8,
    patch_merge_layer = 6,        # at which transformer layer to do patch merging
    patch_merge_num_tokens = 8,   # the output number of tokens from the patch merge
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(4, 3, 256, 256)

preds = v(img) # (4, 1000)
```

One can also use the `PatchMerger` module by itself

```python
import torch
from vit_pytorch.vit_with_patch_merger import PatchMerger

merger = PatchMerger(
    dim = 1024,
    num_tokens_out = 8   # output number of tokens
)

features = torch.randn(4, 256, 1024) # (batch, num tokens, dimension)

out = merger(features) # (4, 8, 1024)
```

## Vision Transformer for Small Datasets

<img src="./images/vit_for_small_datasets.png" width="400px"></img>

This <a href="https://arxiv.org/abs/2112.13492">paper</a>提出了一个新的 图像到块（image-to-patch） 函数，
在对图像进行标准化和划分为 patch（块）之前，
会先对图像进行平移（shift）操作。

作者指出，这种平移操作在其他 Transformer 相关研究中非常有帮助，
因此决定将其纳入以便进一步探索。

此外，该方法还包含了 LSA（局部自注意力，Local Self-Attention） 模块，
其中引入了可学习的温度参数（learned temperature），
并在注意力计算中屏蔽掉 token 对自身的注意（self-attention masking）。



```python
import torch
from vit_pytorch.vit_for_small_dataset import ViT

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(4, 3, 256, 256)

preds = v(img) # (1, 1000)
```

你也可以将这篇论文中的 SPT（Shifted Patch Tokenization，平移块标记化） 作为一个独立模块来使用。

```python
import torch
from vit_pytorch.vit_for_small_dataset import SPT

spt = SPT(
    dim = 1024,
    patch_size = 16,
    channels = 3
)

img = torch.randn(4, 3, 256, 256)

tokens = spt(img) # (4, 256, 1024)
```

## 3D ViT

根据大家的广泛要求，我将开始把本仓库中的部分模型架构扩展为 3D ViT（3D 视觉 Transformer），
以便用于视频、医学影像等三维数据场景。

在使用时，你需要额外传入两个超参数：

frames —— 视频帧的数量；

frame_patch_size —— 在时间维度上（帧方向）的 patch 大小。

以下是一个 3D ViT 的示例起点。

```python
import torch
from vit_pytorch.vit_3d import ViT

v = ViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)
```

3D Simple ViT

```python
import torch
from vit_pytorch.simple_vit_3d import SimpleViT

v = SimpleViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)
```

3D version of <a href="https://github.com/lucidrains/vit-pytorch#cct">CCT</a>

```python
import torch
from vit_pytorch.cct_3d import CCT

cct = CCT(
    img_size = 224,
    num_frames = 8,
    embedding_dim = 384,
    n_conv_layers = 2,
    frame_kernel_size = 3,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 14,
    num_heads = 6,
    mlp_ratio = 3.,
    num_classes = 1000,
    positional_embedding = 'learnable'
)

video = torch.randn(1, 3, 8, 224, 224) # (batch, channels, frames, height, width)
pred = cct(video)
```

## ViViT

<img src="./images/vivit.png" width="350px"></img>

 <a href="https://arxiv.org/abs/2103.15691">paper</a> 提出了三种不同类型的高效视频注意力结构，
其核心思想是将注意力机制在空间（space）和时间（time）维度上进行因式分解（factorization）。

本仓库实现了其中两种变体：

Factorized Encoder（分解式编码器）：先进行空间 Transformer，再进行时间 Transformer；

Factorized Self-Attention（分解式自注意力）：是一个时空 Transformer（spatio-temporal transformer），
其中空间注意力层与时间注意力层交替堆叠。

```python
import torch
from vit_pytorch.vivit import ViT

v = ViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    spatial_depth = 6,         # depth of the spatial transformer
    temporal_depth = 6,        # depth of the temporal transformer
    heads = 8,
    mlp_dim = 2048,
    variant = 'factorized_encoder', # or 'factorized_self_attention'
)

video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)
```

## Parallel ViT

<img src="./images/parallel-vit.png" width="350px"></img>

 <a href="https://arxiv.org/abs/2203.09795">paper</a> 提出在每一层中并行化多个注意力（attention）和前馈网络（feedforward）模块（例如 2 个并行块），
作者声称这种结构在不降低性能的前提下，可以让模型训练得更容易、更稳定。

你可以按照下面的方式尝试这种变体。

```python
import torch
from vit_pytorch.parallel_vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    num_parallel_branches = 2,  # in paper, they claimed 2 was optimal
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(4, 3, 256, 256)

preds = v(img) # (4, 1000)
```

## Learnable Memory ViT

<img src="./images/learnable-memory-vit.png" width="350px"></img>

<a href="https://arxiv.org/abs/2203.15243">paper</a> 表明，在视觉 Transformer（ViT）的每一层中加入可学习的记忆 token（learnable memory tokens），
可以显著提升模型在微调（fine-tuning）阶段的效果。

此外，作者还结合了可学习的任务特定 CLS token 和 适配器头（adapter head），进一步增强了模型性能。

你可以使用下面特别修改过的 ViT 来实现这一机制。

```python
import torch
from vit_pytorch.learnable_memory_vit import ViT, Adapter

# normal base ViT

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(4, 3, 256, 256)
logits = v(img) # (4, 1000)

# do your usual training with ViT
# ...


# then, to finetune, just pass the ViT into the Adapter class
# you can do this for multiple Adapters, as shown below

adapter1 = Adapter(
    vit = v,
    num_classes = 2,               # number of output classes for this specific task
    num_memories_per_layer = 5     # number of learnable memories per layer, 10 was sufficient in paper
)

logits1 = adapter1(img) # (4, 2) - predict 2 classes off frozen ViT backbone with learnable memories and task specific head

# yet another task to finetune on, this time with 4 classes

adapter2 = Adapter(
    vit = v,
    num_classes = 4,
    num_memories_per_layer = 10
)

logits2 = adapter2(img) # (4, 4) - predict 4 classes off frozen ViT backbone with learnable memories and task specific head

```

## Dino

<img src="./images/dino.png" width="350px"></img>

你可以使用最近的自监督学习领域的最新 SOTA 技术——DINO, <a href="https://arxiv.org/abs/2104.14294">Dino</a>, 来训练 ViT 模型，方法如下所示。

<a href="https://www.youtube.com/watch?v=h3ij3F3cPIk">Yannic Kilcher</a> video

```python
import torch
from vit_pytorch import ViT, Dino

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

learner = Dino(
    model,
    image_size = 256,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
)

opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of teacher encoder and teacher centers

# save your improved network
torch.save(model.state_dict(), './pretrained-net.pt')
```

## EsViT

<img src="./images/esvit.png" width="350px"></img>

<a href="https://arxiv.org/abs/2106.09785">`EsViT`</a> 是一种基于上文提到的 DINO 改进而来的自监督学习方法。
它针对带有 patch 合并 / 下采样机制的高效 Vision Transformer（ViT） 进行了重新设计，
在训练时额外引入了一个 区域级损失（regional loss），用于对齐不同增强视图（augmented views）之间的特征。

引用论文摘要中的原话，它在 18 个数据集中的 17 个上都超越了有监督模型的表现，
而且训练吞吐量提升了 3 倍。

尽管名字中包含 “ViT”，但它实际上并不是一种新的 ViT 结构，
而是一种适用于多阶段（multi-stage）ViT 的训练策略。
论文中主要针对 Swin Transformer 进行了研究。

下面的示例展示了如何将 EsViT 应用于 CvT 模型。
你需要将参数 hidden_layer 设置为模型中输出未经过全局池化（global pooling）和投影（projection to logits）前的
那一层的名称，以便获取非平均池化的视觉特征表示。

```python
import torch
from vit_pytorch.cvt import CvT
from vit_pytorch.es_vit import EsViTTrainer

cvt = CvT(
    num_classes = 1000,
    s1_emb_dim = 64,
    s1_emb_kernel = 7,
    s1_emb_stride = 4,
    s1_proj_kernel = 3,
    s1_kv_proj_stride = 2,
    s1_heads = 1,
    s1_depth = 1,
    s1_mlp_mult = 4,
    s2_emb_dim = 192,
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 4,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.
)

learner = EsViTTrainer(
    cvt,
    image_size = 256,
    hidden_layer = 'layers',           # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
)

opt = torch.optim.AdamW(learner.parameters(), lr = 3e-4)

def sample_unlabelled_images():
    return torch.randn(8, 3, 256, 256)

for _ in range(1000):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of teacher encoder and teacher centers

# save your improved network
torch.save(cvt.state_dict(), './pretrained-net.pt')
```

## Accessing Attention

如果你想在研究中可视化注意力权重（softmax 之后的结果），只需按照下面的步骤进行即可。

```python
import torch
from vit_pytorch.vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# import Recorder and wrap the ViT

from vit_pytorch.recorder import Recorder
v = Recorder(v)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 256, 256)
preds, attns = v(img)

# there is one extra patch due to the CLS token

attns # (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)
```

to cleanup the class and the hooks once you have collected enough data

```python
v = v.eject()  # wrapper is discarded and original ViT instance is returned
```

## Accessing Embeddings

你也可以使用 `Extractor` 封装器（wrapper）以类似的方式获取模型的特征嵌入（embeddings）。

```python
import torch
from vit_pytorch.vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# import Recorder and wrap the ViT

from vit_pytorch.extractor import Extractor
v = Extractor(v)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 256, 256)
logits, embeddings = v(img)

# there is one extra token due to the CLS token

embeddings # (1, 65, 1024) - (batch x patches x model dim)
```

Or say for `CrossViT`, which has a multi-scale encoder that outputs two sets of embeddings for 'large' and 'small' scales

```python
import torch
from vit_pytorch.cross_vit import CrossViT

v = CrossViT(
    image_size = 256,
    num_classes = 1000,
    depth = 4,
    sm_dim = 192,
    sm_patch_size = 16,
    sm_enc_depth = 2,
    sm_enc_heads = 8,
    sm_enc_mlp_dim = 2048,
    lg_dim = 384,
    lg_patch_size = 64,
    lg_enc_depth = 3,
    lg_enc_heads = 8,
    lg_enc_mlp_dim = 2048,
    cross_attn_depth = 2,
    cross_attn_heads = 8,
    dropout = 0.1,
    emb_dropout = 0.1
)

# wrap the CrossViT

from vit_pytorch.extractor import Extractor
v = Extractor(v, layer_name = 'multi_scale_encoder') # take embedding coming from the output of multi-scale-encoder

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 256, 256)
logits, embeddings = v(img)

# there is one extra token due to the CLS token

embeddings # ((1, 257, 192), (1, 17, 384)) - (batch x patches x dimension) <- large and small scales respectively
```

## Research Ideas

### Efficient Attention

可能有一些来自计算机视觉领域的研究者仍然认为注意力机制的计算代价是二次复杂度（quadratic cost）。
幸运的是，现在已经有许多新的技术可以缓解这一问题。
本仓库提供了一种方式，让你可以插入（plugin）自己实现的稀疏注意力 Transformer。

以下是一个使用 <a href="https://arxiv.org/abs/2102.03902">Nystromformer</a>的示例。

```bash
$ pip install nystrom-attention
```

```python
import torch
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer

efficient_transformer = Nystromformer(
    dim = 512,
    depth = 12,
    heads = 8,
    num_landmarks = 256
)

v = ViT(
    dim = 512,
    image_size = 2048,
    patch_size = 32,
    num_classes = 1000,
    transformer = efficient_transformer
)

img = torch.randn(1, 3, 2048, 2048) # your high resolution picture
v(img) # (1, 1000)
```

我强烈推荐的其他稀疏注意力框架包括： <a href="https://github.com/lucidrains/routing-transformer">Routing Transformer</a> 和 <a href="https://github.com/lucidrains/sinkhorn-transformer">Sinkhorn Transformer</a>，它们都是非常优秀的稀疏注意力（Sparse Attention）实现，可用于在保持性能的同时显著降低计算复杂度。

### Combining with other Transformer improvements

这篇论文特意使用了最基础版本（最原始、未经改进）的注意力网络，以此来表达其核心观点。
如果你希望使用更先进的注意力网络改进版本，请使用来自这个 <a href="https://github.com/lucidrains/x-transformers">this repository</a>的`Encoder`模块。

例如：

```bash
$ pip install x-transformers
```

```python
import torch
from vit_pytorch.efficient import ViT
from x_transformers import Encoder

v = ViT(
    dim = 512,
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    transformer = Encoder(
        dim = 512,                  # set to be the same as the wrapper
        depth = 12,
        heads = 8,
        ff_glu = True,              # ex. feed forward GLU variant https://arxiv.org/abs/2002.05202
        residual_attn = True        # ex. residual attention https://arxiv.org/abs/2012.11747
    )
)

img = torch.randn(1, 3, 224, 224)
v(img) # (1, 1000)
```

## FAQ

- 如何输入非正方形图像？

你其实已经可以直接输入非正方形图像了——
只需要确保：

图像的高度（height）和宽度（width）都 小于或等于 image_size；

并且 高度和宽度都能被 patch_size 整除。

例如：

```python
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 128) # <-- not a square

preds = v(img) # (1, 1000)
```

- How do I pass in non-square patches?

```python
import torch
from vit_pytorch import ViT

v = ViT(
    num_classes = 1000,
    image_size = (256, 128),  # image size is a tuple of (height, width)
    patch_size = (32, 16),    # patch size is a tuple of (height, width)
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 128)

preds = v(img)
```

## 资源

如果你来自计算机视觉领域，并且是第一次接触 Transformer，下面这些资源能极大地帮助你快速入门：

1. <a href="http://jalammar.github.io/illustrated-transformer/">Illustrated Transformer</a> - Jay Alammar

1. 直观生动地讲解了 Transformer 的结构和工作原理，非常适合初学者。

2. <a href="http://peterbloem.nl/blog/transformers">Transformers from Scratch</a>  - Peter Bloem

2. 从零开始一步步推导 Transformer 的数学原理与实现细节。

3. <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a> - Harvard NLP

3.基于 PyTorch 的可执行讲解版 Transformer，实现、注释和理论结合，非常系统。


## Citations
```bibtex
@article{hassani2021escaping,
    title   = {Escaping the Big Data Paradigm with Compact Transformers},
    author  = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
    year    = 2021,
    url     = {https://arxiv.org/abs/2104.05704},
    eprint  = {2104.05704},
    archiveprefix = {arXiv},
    primaryclass = {cs.CV}
}
```

```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{touvron2020training,
    title   = {Training data-efficient image transformers & distillation through attention}, 
    author  = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Hervé Jégou},
    year    = {2020},
    eprint  = {2012.12877},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{yuan2021tokenstotoken,
    title   = {Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet},
    author  = {Li Yuan and Yunpeng Chen and Tao Wang and Weihao Yu and Yujun Shi and Francis EH Tay and Jiashi Feng and Shuicheng Yan},
    year    = {2021},
    eprint  = {2101.11986},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{zhou2021deepvit,
    title   = {DeepViT: Towards Deeper Vision Transformer},
    author  = {Daquan Zhou and Bingyi Kang and Xiaojie Jin and Linjie Yang and Xiaochen Lian and Qibin Hou and Jiashi Feng},
    year    = {2021},
    eprint  = {2103.11886},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{touvron2021going,
    title   = {Going deeper with Image Transformers}, 
    author  = {Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Hervé Jégou},
    year    = {2021},
    eprint  = {2103.17239},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{chen2021crossvit,
    title   = {CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification},
    author  = {Chun-Fu Chen and Quanfu Fan and Rameswar Panda},
    year    = {2021},
    eprint  = {2103.14899},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{wu2021cvt,
    title   = {CvT: Introducing Convolutions to Vision Transformers},
    author  = {Haiping Wu and Bin Xiao and Noel Codella and Mengchen Liu and Xiyang Dai and Lu Yuan and Lei Zhang},
    year    = {2021},
    eprint  = {2103.15808},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{heo2021rethinking,
    title   = {Rethinking Spatial Dimensions of Vision Transformers}, 
    author  = {Byeongho Heo and Sangdoo Yun and Dongyoon Han and Sanghyuk Chun and Junsuk Choe and Seong Joon Oh},
    year    = {2021},
    eprint  = {2103.16302},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{graham2021levit,
    title   = {LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference},
    author  = {Ben Graham and Alaaeldin El-Nouby and Hugo Touvron and Pierre Stock and Armand Joulin and Hervé Jégou and Matthijs Douze},
    year    = {2021},
    eprint  = {2104.01136},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{li2021localvit,
    title   = {LocalViT: Bringing Locality to Vision Transformers},
    author  = {Yawei Li and Kai Zhang and Jiezhang Cao and Radu Timofte and Luc Van Gool},
    year    = {2021},
    eprint  = {2104.05707},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{chu2021twins,
    title   = {Twins: Revisiting Spatial Attention Design in Vision Transformers},
    author  = {Xiangxiang Chu and Zhi Tian and Yuqing Wang and Bo Zhang and Haibing Ren and Xiaolin Wei and Huaxia Xia and Chunhua Shen},
    year    = {2021},
    eprint  = {2104.13840},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@misc{zhang2021aggregating,
    title   = {Aggregating Nested Transformers},
    author  = {Zizhao Zhang and Han Zhang and Long Zhao and Ting Chen and Tomas Pfister},
    year    = {2021},
    eprint  = {2105.12723},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{chen2021regionvit,
    title   = {RegionViT: Regional-to-Local Attention for Vision Transformers}, 
    author  = {Chun-Fu Chen and Rameswar Panda and Quanfu Fan},
    year    = {2021},
    eprint  = {2106.02689},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{wang2021crossformer,
    title   = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention}, 
    author  = {Wenxiao Wang and Lu Yao and Long Chen and Binbin Lin and Deng Cai and Xiaofei He and Wei Liu},
    year    = {2021},
    eprint  = {2108.00154},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{caron2021emerging,
    title   = {Emerging Properties in Self-Supervised Vision Transformers},
    author  = {Mathilde Caron and Hugo Touvron and Ishan Misra and Hervé Jégou and Julien Mairal and Piotr Bojanowski and Armand Joulin},
    year    = {2021},
    eprint  = {2104.14294},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{he2021masked,
    title   = {Masked Autoencoders Are Scalable Vision Learners}, 
    author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Dollár and Ross Girshick},
    year    = {2021},
    eprint  = {2111.06377},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{xie2021simmim,
    title   = {SimMIM: A Simple Framework for Masked Image Modeling}, 
    author  = {Zhenda Xie and Zheng Zhang and Yue Cao and Yutong Lin and Jianmin Bao and Zhuliang Yao and Qi Dai and Han Hu},
    year    = {2021},
    eprint  = {2111.09886},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{fayyaz2021ats,
    title   = {ATS: Adaptive Token Sampling For Efficient Vision Transformers},
    author  = {Mohsen Fayyaz and Soroush Abbasi Kouhpayegani and Farnoush Rezaei Jafari and Eric Sommerlade and Hamid Reza Vaezi Joze and Hamed Pirsiavash and Juergen Gall},
    year    = {2021},
    eprint  = {2111.15667},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{mehta2021mobilevit,
    title   = {MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
    author  = {Sachin Mehta and Mohammad Rastegari},
    year    = {2021},
    eprint  = {2110.02178},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{lee2021vision,
    title   = {Vision Transformer for Small-Size Datasets}, 
    author  = {Seung Hoon Lee and Seunghyun Lee and Byung Cheol Song},
    year    = {2021},
    eprint  = {2112.13492},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{renggli2022learning,
    title   = {Learning to Merge Tokens in Vision Transformers},
    author  = {Cedric Renggli and André Susano Pinto and Neil Houlsby and Basil Mustafa and Joan Puigcerver and Carlos Riquelme},
    year    = {2022},
    eprint  = {2202.12015},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{yang2022scalablevit,
    title   = {ScalableViT: Rethinking the Context-oriented Generalization of Vision Transformer}, 
    author  = {Rui Yang and Hailong Ma and Jie Wu and Yansong Tang and Xuefeng Xiao and Min Zheng and Xiu Li},
    year    = {2022},
    eprint  = {2203.10790},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{Touvron2022ThreeTE,
    title   = {Three things everyone should know about Vision Transformers},
    author  = {Hugo Touvron and Matthieu Cord and Alaaeldin El-Nouby and Jakob Verbeek and Herv'e J'egou},
    year    = {2022}
}
```

```bibtex
@inproceedings{Sandler2022FinetuningIT,
    title   = {Fine-tuning Image Transformers using Learnable Memory},
    author  = {Mark Sandler and Andrey Zhmoginov and Max Vladymyrov and Andrew Jackson},
    year    = {2022}
}
```

```bibtex
@inproceedings{Li2022SepViTSV,
    title   = {SepViT: Separable Vision Transformer},
    author  = {Wei Li and Xing Wang and Xin Xia and Jie Wu and Xuefeng Xiao and Minghang Zheng and Shiping Wen},
    year    = {2022}
}
```

```bibtex
@inproceedings{Tu2022MaxViTMV,
    title   = {MaxViT: Multi-Axis Vision Transformer},
    author  = {Zhengzhong Tu and Hossein Talebi and Han Zhang and Feng Yang and Peyman Milanfar and Alan Conrad Bovik and Yinxiao Li},
    year    = {2022}
}
```

```bibtex
@article{Li2021EfficientSV,
    title   = {Efficient Self-supervised Vision Transformers for Representation Learning},
    author  = {Chunyuan Li and Jianwei Yang and Pengchuan Zhang and Mei Gao and Bin Xiao and Xiyang Dai and Lu Yuan and Jianfeng Gao},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2106.09785}
}
```

```bibtex
@misc{Beyer2022BetterPlainViT
    title     = {Better plain ViT baselines for ImageNet-1k},
    author    = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
    publisher = {arXiv},
    year      = {2022}
}

```

```bibtex
@article{Arnab2021ViViTAV,
    title   = {ViViT: A Video Vision Transformer},
    author  = {Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lucic and Cordelia Schmid},
    journal = {2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
    year    = {2021},
    pages   = {6816-6826}
}
```

```bibtex
@article{Liu2022PatchDropoutEV,
    title   = {PatchDropout: Economizing Vision Transformers Using Patch Dropout},
    author  = {Yue Liu and Christos Matsoukas and Fredrik Strand and Hossein Azizpour and Kevin Smith},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.07220}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

```bibtex
@inproceedings{Dehghani2023PatchNP,
    title   = {Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution},
    author  = {Mostafa Dehghani and Basil Mustafa and Josip Djolonga and Jonathan Heek and Matthias Minderer and Mathilde Caron and Andreas Steiner and Joan Puigcerver and Robert Geirhos and Ibrahim M. Alabdulmohsin and Avital Oliver and Piotr Padlewski and Alexey A. Gritsenko and Mario Luvci'c and Neil Houlsby},
    year    = {2023}
}
```

```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@inproceedings{Darcet2023VisionTN,
    title   = {Vision Transformers Need Registers},
    author  = {Timoth'ee Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:263134283}
}
```

```bibtex
@inproceedings{ElNouby2021XCiTCI,
    title   = {XCiT: Cross-Covariance Image Transformers},
    author  = {Alaaeldin El-Nouby and Hugo Touvron and Mathilde Caron and Piotr Bojanowski and Matthijs Douze and Armand Joulin and Ivan Laptev and Natalia Neverova and Gabriel Synnaeve and Jakob Verbeek and Herv{\'e} J{\'e}gou},
    booktitle = {Neural Information Processing Systems},
    year    = {2021},
    url     = {https://api.semanticscholar.org/CorpusID:235458262}
}
```

```bibtex
@inproceedings{Koner2024LookupViTCV,
    title   = {LookupViT: Compressing visual information to a limited number of tokens},
    author  = {Rajat Koner and Gagan Jain and Prateek Jain and Volker Tresp and Sujoy Paul},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:271244592}
}
```

```bibtex
@article{Bao2022AllAW,
    title   = {All are Worth Words: A ViT Backbone for Diffusion Models},
    author  = {Fan Bao and Shen Nie and Kaiwen Xue and Yue Cao and Chongxuan Li and Hang Su and Jun Zhu},
    journal = {2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022},
    pages   = {22669-22679},
    url     = {https://api.semanticscholar.org/CorpusID:253581703}
}
```

```bibtex
@misc{Rubin2024,
    author  = {Ohad Rubin},
    url     = {https://medium.com/@ohadrubin/exploring-weight-decay-in-layer-normalization-challenges-and-a-reparameterization-solution-ad4d12c24950}
}
```

```bibtex
@inproceedings{Loshchilov2024nGPTNT,
    title   = {nGPT: Normalized Transformer with Representation Learning on the Hypersphere},
    author  = {Ilya Loshchilov and Cheng-Ping Hsieh and Simeng Sun and Boris Ginsburg},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273026160}
}
```

```bibtex
@inproceedings{Liu2017DeepHL,
    title   = {Deep Hyperspherical Learning},
    author  = {Weiyang Liu and Yanming Zhang and Xingguo Li and Zhen Liu and Bo Dai and Tuo Zhao and Le Song},
    booktitle = {Neural Information Processing Systems},
    year    = {2017},
    url     = {https://api.semanticscholar.org/CorpusID:5104558}
}
```

```bibtex
@inproceedings{Zhou2024ValueRL,
    title   = {Value Residual Learning For Alleviating Attention Concentration In Transformers},
    author  = {Zhanchao Zhou and Tianyi Wu and Zhiyun Jiang and Zhenzhong Lan},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273532030}
}
```

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```

```bibtex
@inproceedings{Fuller2025SimplerFV,
    title   = {Simpler Fast Vision Transformers with a Jumbo CLS Token},
    author  = {Anthony Fuller and Yousef Yassin and Daniel G. Kyrollos and Evan Shelhamer and James R. Green},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276557720}
}
```

```bibtex
@misc{xiong2025ndrope,
    author = {Jerry Xiong},
    title  = {On n-dimensional rotary positional embeddings},
    year   = {2025},
    url    = {https://jerryxio.ng/posts/nd-rope/}
}
```

```bibtex
@inproceedings{anonymous2025vat,
    title   = {{VAT}: Vision Action Transformer by Unlocking Full Representation of ViT},
    author  = {Anonymous},
    booktitle = {Submitted to The Fourteenth International Conference on Learning Representations},
    year    = {2025},
    url     = {https://openreview.net/forum?id=TalHOvvLZu},
    note    = {under review}
}
```

```bibtex
@misc{carrigg2025decorrelationspeedsvisiontransformers,
    title   = {Decorrelation Speeds Up Vision Transformers}, 
    author  = {Kieran Carrigg and Rob van Gastel and Melda Yeghaian and Sander Dalm and Faysal Boughorbel and Marcel van Gerven},
    year    = {2025},
    eprint  = {2510.14657},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2510.14657}, 
}
```

*I visualise a time when we will be to robots what dogs are to humans, and I’m rooting for the machines.* — Claude Shannon
