# Dataset Card for DiscriBench

DiscriBench is a multiple-choice VQA dataset containing 1200 samples to evaluate, analyze, and diagnose the discriminative capability of Vision-Language Models (VLMs).

## Dataset Details

- **Language(s):** English, Japanese
- **License:**  For the Exam portion, the copyright of all the images (`images/{001..033}/*.jpg` and `masked*/*/*.jpg`) and the textual contents (`context`, `question`, and `answer` of qid prefix `001` to `033` in `discribench_en_200.jsonl`, `discribench_ja_200.jsonl`, `ablation/answer_embed_en_100.jsonl`, `ablation/answer_embed_ja_100.jsonl`, `ablation/main_en_100.jsonl`, `ablation/main_ja_100.jsonl`, `ablation/vis_easy_en_100.jsonl`, `ablation/vis_easy_ja_100.jsonl`, `ablation/vis_medium_en_100.jsonl`, and `ablation/vis_medium_ja_100.jsonl`) is all reserved by its original creator, the National Center for University Entrance Examinations (https://www.dnc.ac.jp/). If you want to use this portion, see the [instruction](#use-of-exam-part) below. Images in the COCO portion have separete licenses (See [metadata.jsonl](images/metadata.jsonl)). For the rest part, Apache-2.0 applies.

## Uses

### Direct Use

This dataset is expected to be used to evaluate, analyze, and diagnose the discriminative capability of VLMs.

### Out-of-Scope Use

This dataset is not expected to be used for any purpose other than the above.

## Dataset Structure

```text
- README.md
- discribench_en_200.jsonl: 200 test cases in English (*)
- discribench_ja_200.jsonl: 200 test cases in Japanese (*)
- ablation/*_100.jsonl: 100 test cases each used for ablation study (*)
- images/
    - XXX/*.jpg: image(s) corresponding to question ID (qid), XXX
    - metadata.jsonl: image metadata containing source URLs and individual image licenses
```

(*) Each test case is formatted as follows:

```text
{
qid: question_id[int],
image_files: List[path_to_image_file[str]],
context: situation_and_conversation[str],
question: question[str],
answer: answer[int],
}
```

## Dataset Creation

Please refer to our [papers](##Citation) (to appear) for detail.

### Data Collection and Processing

- Exam: Samples with illustrations are collected from the English listening subject in the Common Test for University Admissions (https://www.dnc.ac.jp/kyotsu/) in Japan, and modified with the following procedures.

  **Details**

  The situation description given in Japanese was machine-translated into English and then manually verified them.
  The speaker names in conversations found in the transcript file were modified: from "M:" and "W:" to "Man:" and "Woman:", respectively.
  The machine-translated English situation description was next concatenated with the associated conversation and question obtained from the ground-truth transcript data.
  The option images were extracted and converted from PDF into JPEG, and resized to be 1280x1280.

- COCO: Samples with natural images are collected from COCO [\[Lin et al., 2014\]](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) using erroneous agreement [\[Tong et al., 2024\]](https://openaccess.thecvf.com/content/CVPR2024/html/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.html).

### Annotation process

We use Claude 3.5 Sonnet (`20240620`) to generate a prototype of the situation description, conversation, question, and answer for COCO samples. We manually verify and modify if necessary the prototype. Please comply with the [Consumer Terms of Service](https://www.anthropic.com/legal/consumer-terms) provided by Anthropic.

### Personal and Sensitive Information

Faces in images are masked to avoid identifying personal information from them.
Texts are manually verified not to have personal or sensitive information.

## Bias, Risks, and Limitations

The language supported in this dataset is English and the images obtained from COCO may implicitly represent subsets of specific countries/regions/areas where the images are taken (e.g., western countries).
These implicit biases might affect the performance of VLMs evaluated on this dataset, and the results should be carefully handled and understood.

## Acknowledgment

These research results were obtained from the commissioned research (No.22501) by National Institute of Information and Communications Technology (NICT), Japan.
The use of the exam data in the English listening subject in the Common Test for University Admissions was granted by the National Center for University Entrance Examinations.

## Citation

If you find our work helpful, please feel free to cite these papers.

```bibtex
@inproceedings{muraoka_nlp2025_discribench,
  title={視覚言語モデルの識別性能に関する評価用ベンチマークの構築},
  author={村岡 雅康, 岡崎 直観},
  booktitle={言語処理学会第31回年次大会 (NLP2025)},
  pages={1196--1201},
  year={2025},
  url={https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/Q3-4.pdf}
}

@inproceedings{muraoka:LREC2026:discribench,
    title = "Evaluating Discriminability of Vision-Language Models",
    author = "Muraoka, Masayasu  and Okazaki, Naoaki",
    booktitle = "The Fifteenth Language Resources and Evaluation Conference",
    month = May,
    year = "2026",
    address = "Palma, Mallorca, Spain",
    url = "To appear"
}
```

## Use of Exam part

If you want to use, modify, redistribute (but not limited to these) the Exam part that the original creator, the National Center for University Entrance Examinations (NCUEE, https://www.dnc.ac.jp/), preserves all the rights, please contact NCUEE to obtain the grant of use regarding the copyright if necessary. Please also refer to the following site or relevant information about copyright in Japan when you consider the necessity of the grant use from NCUEE.

- Reference (in Japanese): https://www.bunka.go.jp/seisaku/chosakuken/seidokaisetsu/index.html, Copyright Textbook for Fiscal Year 2025 (Agency for Cultural Affairs, Government of Japan), 令和7年度著作権テキスト（文化庁）.
- Reference (in English): https://www.bunka.go.jp/english/policy/copyright/pdf/94055801_01.pdf, "General Understanding on AI and Copyright in Japan" - Overview (published by the Legal Subcommittee under the Copyright Subdivision of the Cultural Council).
- Contact (in Japanese): https://www.dnc.ac.jp/about_site.html#mondai

## Dataset Card Contact

Please contact us (masayasu.muraoka@nlp.comp.isct.ac.jp) for any inqueries.
