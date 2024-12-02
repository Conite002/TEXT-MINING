# TEXT MINING

* Classez les documents selon le sentiment, topics, opinion etc
* Application:

    Google translate
        * seq2seq -> detection de tags ou type grammatical

Taches du text Mining

* Processus du text mining

* Corpus : enseble de documents 

WSD : Word Sence Desembiguition
World Net
Optimisation : Pruning -> Elimination des variable non important, 

## Pretraitements 

 `NLP` :  Obtenir a la fin une representation de donnees qui est plus ou moins structures pour une machine
    * Classification ou Clustering
 **Etape**
 corpus -> OCR | Speech Recgintion | etc.. -> NLP -> Extraction(sentiments)


 Les taches du NLP traitent :
* Segmentation (Tokenisation)
* Analyse morphologique : type canonique, radicale


* Etiquetage morpho syntaxique pos tagging
* Analyse semantique
* Analyse 

Quelques ex d'extractiond'information
    NER (Named Entity Recognition)
    Detection de relation
    Extraction d'evenements
    Analyse temporelle
    Template filling (title, abstract, keys word)

## Traitements
* Decouverte de patterns 
* Decouvertes de tendances
* Categorisations

# Optimisation

## Pretraitements 

### Analyse morphologique (Segmentation)
`Subdivision de texte en tokens`

* Monogram
* Bigrams
* Trigrams
* Ngrams

### Analyse lexicale : radicale + l'ordre grammaticale du mot

### Analyse semantique (Approche lexicale)
### Analyse semantique Statisque


Modele representative :

* OHV, BOW, TFIDF
Ont tous un incovenient en commun

## Word 2 Vect

Matrice de co-occurence
decomposition en SVD

Matrice generalement creuse

Solution


Representation vectoriel Terms du context(n) - Terms Centrale
Skyp gram
methodes 


W2V vs SVD
Probabilité    || Statistics; dificille dinserer de nouveaux mots, pas bon pour les grands matrices

W2V -> Glove

Une fois les representations vectorielle obtenue, passons aux operations vectoriels

- Notions de similarites
- Notions de distances ( euclidiennes\, Indice de Jaccard, Similarite de Cosinus)

**Classification**
- Mono class
- Multi class

**Indexation** 
    On utlise des mots cles poue retrouver les document 

**Application : Tri de Documents**
**Application : Filtrage de Documents**
**Modeles de classification**
* Naive bayes
* Logistic regression**
**Decision tree** 
**Decision rule** tout les features appartiennent au document puis on passe a eliminiation
**KNN** Les documents voisin (a base de caracteristique) decideront de la classe

**Bagging and Boosting**

les models plus perfomants : SVM, KNN, RL
Peu perfomants : NN, Decision Tree
Moins performant: Naives bayes

Unite de mesure : L'emission carbone, unite de radiation, interpretabilite,  

**Types de clustering** On fixe de base le nombre de clusters
* Clustering Hearichique

* Clustering  Durs et souples


**Evaluation du clustering