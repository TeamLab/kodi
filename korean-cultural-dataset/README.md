# Korean Cultural Dataset (KCD) for KODI

This dataset is a curated collection of Korean cultural images and text descriptions, specifically designed for training and evaluating Korean-aware diffusion models.

## 📊 Dataset Overview

The Korean Cultural Dataset consists of high-quality Korean images with culturally-aware Korean text captions that reflect various aspects of Korean cultural elements, including heritage structures, landmarks, food, and traditional clothing. Each image is paired with descriptive Korean text captions to enable culturally-appropriate text-to-image generation tasks.

The dataset is provided as a curated collection (`kcd.parquet`) with image references and captions, while the actual images can be downloaded from the official AI Hub data sources listed below.

### Dataset File Structure

- `kcd.parquet`: Main dataset file containing image metadata and Korean cultural captions
- Image files: Available for download from the AI Hub sources referenced in the Data Sources section

### Categories

- **Heritage Structures**
- **Landmarks**
- **Food**
- **Traditional Clothing**


## 🔗 Data Sources

This dataset is compiled from the following official Korean AI Hub datasets:

1. **Korean Food Image Dataset**
   - Source: [AI Hub - Korean Food Image Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=79)
   - Description: Comprehensive collection of Korean food images

2. **Korean Object Image Dataset**
   - Source: [AI Hub - Korean Object Image Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=144)
   - Description: Traditional Korean architecture, artifacts, cultural sites and landmarks

3. **Gyeongbuk World Cultural Heritage Seowon Metaverse Images and 3D Data**
   - Source: [AI Hub - Gyeongbuk World Cultural Heritage Seowon Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71512)
   - Description: images of Heritage Seowons (Dosan, Byeongsan, Sosu, Oksan)

4. **Traditional Korean Hanbok 3D Data**
   - Source: [AI Hub - Traditional Korean Hanbok 3D Data](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71733)
   - Description: Hanbok and traditional Korean clothing with cultural context

### Automated Download and Filtering

*Note: Automated download and filtering scripts for the above data sources are coming soon...*

