# KNDM  

## Introduction  

The project  is an implementation of a knowledge graph transformer and recursive gated meta-path semantic feature learning for prediction of drug-related microbes (KNDM).

---

## Catalogs

- **/data**: Contains the dataset used in our method.
- **data_loader.py**:  Processing drug and microbial similarities and associations, forming embeddings, adjacency matrices, etc.
- **early_stopping.py**:   In order to save better parameters for the model.
- **model.py**: Defines the model.
- **train.py**: Optimize the characterization of microbial (drug) nodes.
- **main.py**: Trains the model.

---

## Original Databases of Dataset Sources



|           File_name            |    Data_type    |                            Source                            |
| :----------------------------: | :-------------: | :----------------------------------------------------------: |
|       drugsimilarity.txt       |    drug-drug    | [Long *et al.*](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i779/6055932?login=false)$^{1}$  |
| microbe_microbe_similarity.txt | microbe-microbe | [Long *et al.*](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i779/6055932?login=false)$^{1}$ |
|            net1.mat            |  drug-microbe   | [Long *et al.*](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i779/6055932?login=false)$^{1}$ |


[1] Long, Y.; Wu, M.; Liu, Y.; Kwoh, C. K.; Luo, J.; Li, X. Ensembling graph attention networks for human microbe–drug association prediction. *Bioinformatics* **2020**, *36*, i779–i786.

## Environment  

The KNDM code has been implemented and tested in the following development environment: 

- Python == 3.9.18
- Matplotlib == 3.9.2
- PyTorch == 2.2.1 
- NumPy ==  1.24.1

---

## Dataset 

- **drug_names.txt**: Contains the names of 1373 drugs.
- **microbe_names.txt**: Contains the names of 173 microbes.
- **drugsimilarity.rar**: A compressed file that contains the following one files.
  - **drugsimilarity.txt**: Contains the drug similarities.
- **microbe_microbe_similarity.txt**: Contains the microbe similarities.  
- **net1.mat**: Represents the adjacency matrix of the drug-microbe heterogeneous graph.
- **Supplemental_Table_ST1.xlsx**: Lists the top 20 candidate microbes for each drug.

---

## How to Run the Code  

1. **Data preprocessing**: Processing drug and microbial similarities and associations, forming embeddings, adjacency matrices, etc.

   ```bash
   python data_loader.py
   ```

2. **Optimize microbial (drug) node features**.

   ```
   python train.py
   ```

   Please create a 'checkpoint' folder to store the model parameters before running the file.

3. **Train and test the model**.  

   ```bash
   python main.py
   ```
