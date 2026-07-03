# 🌿 Frog Sound ID: End-to-End Bioacoustic Pipeline Prototype 

An applied ML engineering showcase demonstrating an end-to-end signal processing and computer vision pipeline for classifying frog vocalizations. This architecture transforms raw 1D audio waveforms into 2D Mel-spectrogram images, handles heavy class imbalances via custom batch sampling, trains a convolutional neural network (CNN), and optimizes the final asset via INT8 quantization for ultra-low latency mobile deployment.

----------------------------------------------
### Top 10 Frog/Toad Species of Southern New Jersey
Here are the top 10 frog and toad species native to Southern New Jersey (and the Pine Barrens) that have substantial acoustic footprints on [xeno-canto](https://xeno-canto.org):

* **Pine Barrens Treefrog** ***(Dryophytes andersonii)*** – The local icon. High conservation value and a distinct, nasal "quank" call.

* **Fowler's Toad** ***(Anaxyrus fowleri)*** – Extremely common in South Jersey sandy soils; makes a nasal, unmusical drone.

* **Bronze Frog** ***(Aquarana clamitans)*** – A.k.a. "Green Frog". Found in almost every pond; sounds exactly like a loose banjo string.

* **American Bullfrog** ***(Aquarana catesbeianus)*** – The classic, deep, resonant "jug-o-rum" bass call.

* **New Jersey Chorus Frog** ***(Pseudacris kalmi)*** – An early spring breeder with a call like running a finger down a plastic comb.

* **Spring Peeper** ***(Pseudacris crucifer)*** – High-pitched, piercing bird-like whistles that dominate early spring nights.

* **Southern Leopard Frog** ***(Lithobates sphenocephalus)*** – Common in coastal plains; makes a distinct chuckling, guttural croak.

* **Wood Frog** ***(Lithobates sylvaticus)*** – Explosive early breeders that sound like a flock of ducks quacking in the woods.

* **Gray Treefrog** ***(Hyla versicolor)*** – Loud, musical, bird-like trills heard throughout summer evenings.

* **Cope's Gray Treefrog** ***(Hyla chrysoscelis)*** – Visually identical to the Gray Treefrog but has a distinctly faster, harsher trill frequency (a perfect test case for ML separation!).

----------------------------------------------
## Dataset & Data Curation

Data Curation & Reproducibility: To respect data hosting limits and maintain a lightweight repository footprint, raw audio assets are not tracked via version control. The dataset utilizes (up to) 10 core indicator species of the Southern New Jersey coastal plain and Pine Barrens ecosystem. Complete metadata, including xeno-canto catalog numbers and recording credits, is provided in ```data/dataset_manifest.csv```. Running python ```src/download_data.py``` will automatically fetch and structure the target audio files.

----------------------------------------------
## 🏗️ Pipeline Architecture

```mermaid
graph TD
    A[Raw 1D Audio .wav file] -->|22.05kHz Resampling| B[STFT Signal Processing]
    B -->|Hann Windowing & Mel-Scale| C[2D Mel-Spectrogram Decibels]
    C -->|Weighted Random Sampler| D[PyTorch EfficientNet / ResNet]
    D -->|Frequency & Time Masking| E[Species Inference Matrix]
    E -->|Post-Training Quantization| F[Edge-Optimized TFLite Engine]
