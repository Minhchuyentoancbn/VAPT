# Adaptive Prompt: Unlocking the Power of Visual Prompt Tuning

This repository is the official implementation of `Adaptive Prompt: Unlocking the Power of Visual Prompt Tuning`.

Visual Prompt Tuning (VPT) has recently emerged as a powerful method for adapting pre-trained vision models to downstream tasks. By introducing learnable prompt tokens as task-specific instructions, VPT effectively guides pre-trained transformer models with minimal overhead. Despite its empirical success, a comprehensive theoretical understanding of VPT remains an active area of research. Building on recent insights into the connection between mixture of experts and prompt-based approaches, we identify a key limitation in VPT: *the restricted functional expressiveness in prompt formulation*. To address this limitation, we propose Visual Adaptive Prompt Tuning (VAPT), a new generation of prompts that redefines prompts as adaptive functions of the input. Our theoretical analysis shows that this simple yet intuitive approach achieves optimal sample efficiency. Empirical results on VTAB-1K and FGVC further demonstrate VAPT’s effectiveness, with performance gains of 7.34\% and 1.04\% over fully fine-tuning baselines, respectively. Notably, VAPT also surpasses VPT by a substantial margin while using fewer parameters. These results highlight both the effectiveness and efficiency of our method and pave the way for future research to explore the potential of adaptive prompts.

## Requirements
- python 3.8.12
- PyTorch 1.7.1
- torchvision 0.8.2
- timm 0.5.4
- CUDA 11.0

## Environment settings
```
conda create -n [ENV_NAME] python=3.8.12 -y
conda activate [ENV_NAME]
bash env_install.sh
```

See `env_setup.sh` if you have any issues with the environment setup. 


## Structure of the this repo (key files are marked with 👉):

- `src/configs`: handles config parameters for the experiments.

  * 👉 `src/config/config.py`: <u>main config setups for experiments and explanation for each of them. </u> 

- `src/data`: loading and setup input datasets. The `src/data/vtab_datasets` are borrowed from 

  [VTAB github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data).


- `src/engine`: main training and eval actions here.

- `src/models`: handles backbone archs and heads for different fine-tuning protocols 

  * 👉`src/models/vit_prompt`: <u>a folder contains the same backbones in `vit_backbones` folder,</u> specified for VAPT. This folder should contain the same file names as those in  `vit_backbones`

  * 👉 `src/models/vit_models.py`: <u>main model for transformer-based models</u> ❗️Note❗️: Current version only support ViT, Swin and ViT with mae, moco-v3

  * `src/models/build_model.py`: main action here to utilize the config and build the model to train / eval.

- `src/solver`: optimization, losses and learning rate schedules.  
- `src/utils`: helper functions for io, loggings, training, visualizations. 
- 👉`train.py`: call this one for training and eval a model with a specified transfer type.
- 👉`tune_fgvc.py`: call this one for tuning learning rate and weight decay for a model with a specified transfer type. We used this script for FGVC tasks.
- 👉`tune_vtab.py`: call this one for tuning vtab tasks: use 800/200 split to find the best lr and wd, and use the best lr/wd for the final runs
- `launch.py`: contains functions used to launch the job.


## Experiments

### Key configs:

- 🔥VAPT related:
  - MODEL.PROMPT.ADAPTIVE: whether to use adaptive prompt
  - MODEL.PROMPT.KERNEL: kernel size of the channel-wise convolution layer
  - MODEL.PROMPT.HIDDEN_DIM: hidden dim of the feature projector
- VPT related:
  - MODEL.PROMPT.NUM_TOKENS: prompt length
  - MODEL.PROMPT.DEEP: deep or shallow prompt
- Fine-tuning method specification:
  - MODEL.TRANSFER_TYPE
- Vision backbones:
  - DATA.FEATURE: specify which representation to use
  - MODEL.TYPE: the general backbone type, e.g., "vit" or "swin"
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints
- Optimization related: 
  - SOLVER.BASE_LR: learning rate for the experiment
  - SOLVER.WEIGHT_DECAY: weight decay value for the experiment
  - DATA.BATCH_SIZE
- Datasets related:
  - DATA.NAME
  - DATA.DATAPATH: where you put the datasets
  - DATA.NUMBER_CLASSES
- Others:
  - RUN_N_TIMES: ensure only run once in case for duplicated submision, not used during vtab runs
  - OUTPUT_DIR: output dir of the final model and logs
  - MODEL.SAVE_CKPT: if set to `True`, will save model ckpts and final output of both val and test set


### Datasets preperation:

- Fine-Grained Visual Classification tasks (FGVC): The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found in `local_datasets` folder.

  - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)

  - [NABirds](http://info.allaboutbirds.org/nabirds/)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

- [Visual Task Adaptation Benchmark](https://google-research.github.io/task_adaptation/) (VTAB): see `VTAB_SETUP.md` for detailed instructions and tips.



### Pre-trained model preperation

Download and place the pre-trained Transformer-based backbones to `MODEL.MODEL_ROOT`. Note that you also need to rename the downloaded ViT-B/16 ckpt from `ViT-B_16.npz` to `imagenet21k_ViT-B_16.npz`.


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">link</a></td>
<td align="center"><tt>d9715d</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">link</a></td>
<td align="center"><tt>8f39ce</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">link</a></td>
<td align="center"><tt>8cad7c</tt></td>
</tr>
</tbody></table>


### Examples

```bash
# Training of VAPT
python tune_vtab.py \
    --train-type "prompt" \
    --config-file configs/prompt/natural/cifar100.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "64" \
    MODEL.PROMPT.NUM_TOKENS "10" \
    MODEL.PROMPT.DEEP "True" \
    MODEL.PROMPT.DROPOUT "0.1" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    OUTPUT_DIR "output/tune_vtab_cifar100_vapt" \
    MODEL.PROMPT.ADAPTIVE "True" \
    MODEL.PROMPT.DROPOUT_MLP "0.1" \
    MODEL.PROMPT.HIDDEN_DIM "8"
```


## License

The majority of VAPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/Minhchuyentoancbn/Visual-Adaptive-Prompt-Tuning/blob/master/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.