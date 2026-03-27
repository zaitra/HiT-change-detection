# Towards Onboard Continuous Change Detection for Floods

This repository contains the official implementation of the **HiT (History-Injection Transformer)** mechanism, as described in the paper: [**"Towards Onboard Continuous Change Detection for Floods."**](https://arxiv.org/html/2601.13751v3)

HiT maintains historical context from previous observations while reducing data storage by over 99% of original image size (compared to bi-temporal baseline). Testing on the STTORM-CD-Floods dataset confirms that the HiT mechanism within the PrithVi-tiny foundation model maintains detection accuracy compared to the bi-temporal baseline.


| Model           | F1          | Precission  | Recall      | Parameters | Input Size |
| --------------- | ----------- | ----------- | ----------- | ---------- | ---------- |
| Baseline        | 0.41 ± 0.06 | 0.73 ± 0.05 | 0.29 ± 0.05 | 8.5M       | 2×         |
| **HiT-PrithVi** | 0.38 ± 0.08 | 0.70 ± 0.03 | 0.27 ± 0.08 | 7.8M       | **1.004×** |
| ContUrbanCD     | 0.46 ± 0.26 | 0.82 ± 0.06 | 0.35 ± 0.25 | 25M        | n×         |

Checkpoints for **HiT-PrithVi** are available on Hugging Face - [link](https://huggingface.co/ZAITRA/HiT-Prithvi).

-----

### 📊 Datasets

- [**STTORM-CD-Floods**](https://zenodo.org/records/14891438)
- [**RaVAEn-Floods**](https://drive.google.com/drive/folders/1VEf49IDYFXGKcfvMsfh33VSiyx5MpHEn)


### Installation

```bash
git clone https://github.com/zaitra/HiT-change-detection.git
cd Hit-change-detection
pip install .
```


## 📜 Citation

If you use this work, please cite the following paper:

```bibtex
@misc{kyselica2026towards,
      title={Towards Onboard Continuous Change Detection for Floods}, 
      author={Daniel Kyselica and Jonáš Herec and Oliver Kutis and Rado Pitoňák},
      year={2026},
      eprint={2601.13751},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.13751}, 
}
```

-----

**Maintained by [Zaitra](https://zaitra.io).**

