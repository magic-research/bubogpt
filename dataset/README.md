# Audio Dataset Instruction

## Stage1: Pretraining 
We mainly use [WavCaps](https://github.com/XinhaoMei/WavCaps) dataset for pre-training. 

### Download 

```Bash
# install git-lfs
sudo apt update
sudo apt-get install git-lfs


git clone https://huggingface.co/datasets/cvssp/WavCaps
cd WavCaps
git lfs pull --include "*" 
```

### Processing

1. Extract zip file
```bash
# merge shards first
zip -s- FILE_NAME.zip -O COMBINED_FILE.zip
unzip COMBINED_FILE.zip
```

2. Processing
Extract raw audio data
```bash
unzip COMBINED_FILE.zip -d /target/dir
```

Create json files (annotations) for each example. Before processing, modify `dataset/audio/process.py` to set data and json path. 
```bash
python3 --dataset test --data_dir /path/to/data --json_path /path/to/json
```


3. Pack with tar
```bash
python3 dataset/audio/make_tar.py --input /path/to/data --output /path/to/web_dataset \
    --dataclass none --filename filename --num_element 500
```

To view tar file 
```
tar tf filename.tar | sed 10q
```

**To setup in one line:**
```bash
# DATASET=soundbible bbc audioset freesound
DATASET=soundbible bash dataset/audio/setup.sh
```


## Stage2: Instruction Tuning
 
We use [Clotho](https://arxiv.org/pdf/1910.09387.pdf) as the base corpus to construct our instruction tuning dataset **Clotho-Detail**. 

### Download
1. Access [Clotho Dataset](https://zenodo.org/record/3490684) to download the [clotho_audio_development.7z](https://zenodo.org/record/3490684/files/clotho_audio_development.7z?download=1) and [clotho_audio_evaluation.7z](https://zenodo.org/record/3490684/files/clotho_audio_evaluation.7z?download=1) audio files. 
2. Download the generated annotation file [Clotho-Detail](https://huggingface.co/datasets/magicr/BuboGPT/blob/main/Clotho-detail-annotation.json).

### Processing
1. Unzip the files above and merge all the audios into a single folder *audio*. As a result, there should be 3,939 audios contained in the folder. 
2. Put the annotation file under the same file hierarchical level as the *audio* folder, like:
```
clotho
├─ Clotho-detail-annotation.json
├─ audio
├─── 00294 harvest festival rumour 1.wav
├─── 00332 lake beach 1.wav
├─── ...
```
3. Edit the path and name configuration in the corresponding files accordingly.


# Image-Audio Dataset Instruction

## Part 1: Aligned Audio-Image Data
We use [VGGSS](https://arxiv.org/pdf/2104.02691.pdf) as the base data to construct our training corpus in the process of multi-modality instruction tuning.

To explore and exploit this corpus, please:
1. Follow the [github page](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way) and [project page](https://www.robots.ox.ac.uk/~vgg/research/lvs/) of VGGSS to prepare the audio and image data into the *audio* and *image* folders.
2. Download our refactored annotation file [VGGSS-Instruction](https://huggingface.co/datasets/magicr/BuboGPT/blob/main/vggss-instruction-tuning.json).
3. Put the annotation file under the same file hierarchical level of *audio* and *image* folders, like:
```
VGGSS
├─ audio
├─── 007P6bFgRCU_000010.wav
├─── 00QQLLcny14_000083.wav
├─ image
├─── 007P6bFgRCU_000010.jpg
├─── 00QQLLcny14_000083.jpg
├─ vggss-instruction-tuning.json
```
4. Edit the path and name configuration in the corresponding files accordingly.

## Part 2: Unaligned Audio-Image Data
The unaligned audio-image data can be collected by pairing arbitrary image and audio data from different datasets. Please refer to the [config](../bubogpt/configs/datasets/aud_img_neg/default.yaml) of negatively paired audio-image dataset and modify the configuration accordingly.
