# TrOCR Marathi Printed


This project is about using TrOCR, a Transformer-based Optical Character Recognition model, to recognize text from scanned Marathi documents. TrOCR is a state-of-the-art model that leverages both computer vision and natural language processing to achieve high accuracy and robustness on various text recognition tasks.

## Dataset

The dataset consists of 2671 PNG images of lines extracted from scanned Marathi documents, and a CSV file `output.csv` that contains two columns: one for the image file name and the other for the corresponding text. The images and the CSV file are compressed in a zip file `output-zip.zip` and [un-preprocessed](https://drive.google.com/drive/folders/1EZ2lc8eZkOFeopAD_3eztsO9-WcCLgvP?usp=sharing) for convenience.

## Notebooks

The project contains two Jupyter notebooks: `train.ipynb` and `test.ipynb`. The train notebook shows [how to fine-tune a pre-trained TrOCR](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb) model on the Marathi dataset using the [Hugging Face Transformers library](https://huggingface.co/docs/transformers/en/index). The test notebook shows how to use the fine-tuned model to perform text recognition on new images and evaluate its performance.

## Usage

To use this project, you need to have Python 3.6 or higher and install the required packages listed in the requirements.txt file. You also need to download the pre-trained TrOCR model from the Hugging Face model hub and save it in the models folder. Then, you can run the notebooks in your preferred environment, such as Google Colab or your local machine.

## Pre-trained Models]

The project uses two pre-trained models for the vision encoder and the text decoder:

- [Google ViT](https://huggingface.co/google/vit-base-patch16-224): A Vision Transformer model that encodes an input image as a sequence of patches and applies self-attention to learn global features.
- [Marathi-BERT-v2](https://huggingface.co/l3cube-pune/marathi-bert-v2): MahaBERT is a Marathi BERT model. It is a multilingual BERT (google/muril-base-cased) model fine-tuned on L3Cube-MahaCorpus and other publicly available Marathi monolingual datasets.

## License

The dataset is created by me and I own the rights to it. If you want to use it for your own research or projects, you must contact me first and obtain my permission. The code and the notebooks are licensed which means you can use, modify, and distribute them freely, as long as you give credit to me and the original sources.

## References

Li, M., Liu, Y., Gao, X., He, Y., Chen, W., Qiao, S., ... & Chen, X. (2021). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. arXiv preprint arXiv:2109.10282.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Brew, J. (2019). Huggingface's transformers: State-of-the-art natural language processing. ArXiv, abs/1910.03771.




## Important Links

1. **Marathi-BERT-V2 by l3cube-pune**  
   Hugging Face Model: [marathi-bert-v2](https://huggingface.co/l3cube-pune/marathi-bert-v2)
   
2. **Marathi-BERT by l3cube-pune**  
   Hugging Face Model: [marathi-bert](https://huggingface.co/l3cube-pune/marathi-bert)
   
3. **Marathi_BERT_V2 BertEmbeddings from l3cube-pune**  
   More Information: [Marathi_BERT_V2 BertEmbeddings](https://sparknlp.org/2023/09/13/marathi_bert_v2_mr.html) by l3cube-pune
   
4. **Vision Transformer by Google Research**  
   GitHub Repository: [vision_transformer](https://github.com/google-research/vision_transformer)
   
5. **Marathi-BERT-V2 by l3cube-pune**  
   Hugging Face Model: [marathi-bert-v2](https://huggingface.co/l3cube-pune/marathi-bert-v2)
   
6. **Vision Transformer by Google Research**  
   GitHub Repository: [vision_transformer](https://github.com/google-research/vision_transformer)
   
7. **VIT-Base-Patch16-224 by Google**  
   Hugging Face Model: [vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
   
8. **L3Cube-MahaCorpus and other publicly available Marathi monolingual datasets**  
   GitHub Repository: [MarathiNLP](https://github.com/l3cube-pune/MarathiNLP)
