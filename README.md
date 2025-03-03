# MV-MATH🔥: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts

![MathQA](https://img.shields.io/badge/Task-MathQA-red) 
![Mathematical Reasoning](https://img.shields.io/badge/Task-Mathematical_Reasoning-red) 
![Multimodal Reasoning](https://img.shields.io/badge/Task-Multi--Modal-red) 

![ChatGPT](https://img.shields.io/badge/Model-ChatGPT-green) 
![GPT-4](https://img.shields.io/badge/Model-GTP--4o-green) 
![GPT-4V](https://img.shields.io/badge/Model-Claude--3.5--Sonnet-green)
![GPT-4V](https://img.shields.io/badge/Model-QvQ-green)
![Gemini](https://img.shields.io/badge/Model-Gemini-green)

🌟  This is the official repository for the paper "[MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts](https://arxiv.org/pdf/2502.20808)", which contains both evaluation code and data for the **MV-MATH** benchmark.

[[🌐 Homepage](https://eternal8080.github.io/MV-MATH.github.io/)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/PeijieWang/MV-MATH)] [[📊 Leaderboard ](https://eternal8080.github.io/MV-MATH.github.io/)] [[🔍 Visualization](https://eternal8080.github.io/MV-MATH.github.io/)] [[📖 ArXiv Paper](https://arxiv.org/abs/2502.20808)]

## 💥 News

- **[2025-03-01]** 🚀🚀🚀 See this page for the [homepage](https://eternal8080.github.io/MV-MATH.github.io/) pf **MV-MATH**
- **[2025-03-01]** 🔥🔥🔥 O1-like model **QVQ-72B-Preview** achieves **29.3%**, establishing itself as the new best-performing open-sourced model. 🎉 Congratulations!
- **[2025-02-27]** Our dataset is now accessible at [huggingface](https://huggingface.co/datasets/PeijieWang/MV-MATH).
- **[2025-02-27]** The top-performing model, **Claude-3.5-Sonnet** only scores **33.9%** on **MV-MATH**, while human performance is around **76%**.
- **[2025-02-27]** **MV-MATH** is accepted by CVPR2025! 🎉🎉🎉

## 👀 Introduction

MV-MATH is a meticulously annotated dataset designed to evaluate the mathematical reasoning capabilities of MLLMs in multi-visual contexts. Each sample in MV-MATH consists of interleaved multi-image and text. It comprises 2,009 multi-image questions, with some questions containing up to 8 images. It includes three types: multiple-choice, free-form and multi-step questions.

MV-MATH is organized into 11 subjects over 3 difficulty levels, including Analytic Geometry, Algebra, Metric Geometry, Combinatorics, Transformation Geometry, Logic, Solid Geometry, Arithmetic, Combinatorial Geometry, Descriptive Geometry and Statistics, covering a range of scenarios from the K-12 mathematics curriculum.

Based on image relevance, we categorize MV-MATH into two subsets: a mutually dependent set (MD), where images are interrelated and understanding one image necessitates information from another; and an independent set (ID), where images are unrelated and can be interpreted independently without reference to other images.


<p align="center">
    <img src="assets/figures/figure1_new.png" width="100%"> <br>
  The accuracies of four prominent Large Multimodal Models (LMMs), random chance, and human
performance are evaluated on our proposed <b>MATH-Vision (MATH-V)</b> across 16 subjects.
</p>

Through extensive experimentation, we unveil a notable performance gap between current LMMs and human performance on MATH-V, underscoring the imperative for further advancements in LMMs.





You can refer to our [project homepage](https://mathllm.github.io/mathvision/) and [the paper](https://arxiv.org/pdf/2402.14804.pdf) for more details.

## 📐 Dataset Examples

Some examples of MATH-V on three subjects: analytic geometry, topology, and graph theory.

<details>
<summary>Analytic geometry</summary><p align="center">
    <img src="assets/examples/exam_analytic_geo.png" width="50%"> <br>
</p></details>

<details>
<summary>Topology</summary><p align="center">
    <img src="assets/examples/exam_topology.png" width="50%"> <br>
</p></details>

<details>
<summary>Graph Geometry</summary><p align="center">
    <img src="assets/examples/exam_graph.png" width="50%"> <br>
</p></details>

You can refer to the Appendix A.4 of [the paper](https://arxiv.org/pdf/2502.20808) for example images of 11 subjects.

## 🏆 Leaderboard

The leaderboard is available [here](https://eternal8080.github.io/MV-MATH.github.io/).



## 📈 Evaluation

### Generating Outputs of Different Models

#### Gemini

`python models/Gemini.py --in_path ./data/test.jsonl --save_path ./Gemini.jsonl`

This will run the Gemini API and save the outputs to `./Gemini.jsonl` path. You can modify the system prompt, max tokens, etc. in the `benchmark_gemini` function.

#### GPT_with_caption

Generate image captions using GPT-4V:

`python models/GPT_with_caption.py --model gpt-4-vision-preview --in_path ./data/test.jsonl --save_path ./data/gpt4v-captions.jsonl`

Generate outputs using ChatGPT-3.5 or GPT-4 with image captions:

`python models/GPT_with_caption.py --model gpt-3.5-turbo-0125 (gpt-4-turbo-preview) --in_path ./data/test.jsonl --save_path ./gpt3.5_caption.jsonl (./gpt4_caption.jsonl)`



### Evaluation of Model Outputs

Once all the model outputs have been generated, execute the `python evaluation/evaluate.py` function to assess these outputs. This script will examine all outputs located in the `outputs/` directory, computing overall accuracy as well as accuracy for each subject and level.

You can refer to the Appendix E and F of [the paper](https://arxiv.org/pdf/2402.14804.pdf) for some evaluation results of the above models.

## 📝 Citation

If you find this benchmark useful in your research, please consider citing this BibTex:

```
@inproceedings{
    wang2024measuring,
    title={Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset},
    author={Ke Wang and Junting Pan and Weikang Shi and Zimu Lu and Houxing Ren and Aojun Zhou and Mingjie Zhan and Hongsheng Li},
    booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2024},
    url={https://openreview.net/forum?id=QWTCcxMpPA}
}
```

## 🧠 Related Work

- **[CSV🔥]** [Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification](https://wangk.org/publications/1_iclr2024_csv/)
- **[MathGenie]** [MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs](https://github.com/MathGenie/MathGenie)
- **[MathCoder🔥]** [MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning](https://github.com/mathllm/MathCoder)
- **[MathVerse]** [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](https://github.com/ZrrSkywalker/MathVerse)
- **[MathVista]** [MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts](https://github.com/lupantech/MathVista)
- **[SPHINX]** [The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal LLMs](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
- **[SPHINX-X]** [Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
