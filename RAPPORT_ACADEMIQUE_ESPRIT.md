# RAPPORT ACADEMIQUE

## PROJET 1 | Analyse Automatique des Sentiments pour Ameliorer l'Experience Client

**Etablissement :** ESPRIT  
**Annee universitaire :** 2024-2025  
**Projet realise par :** Baha Sallami, Mariem Bouchaala  
**Encadrant pedagogique :** Prof. Anis Bel Hadj Hassin  

---

## Page de garde

**Titre du projet**  
PROJET 1 | Analyse Automatique des Sentiments pour Ameliorer l'Experience Client

**Nature du document**  
Rapport academique de realisation d'une application NLP et Web analytique.

**Contexte**  
Ce projet s'inscrit dans le cadre d'un travail academique consacre au traitement automatique du langage naturel, a la fouille d'opinions et a la valorisation de la voix du client dans un contexte e-commerce. L'application developpee permet de collecter, nettoyer, analyser, visualiser et interpreter des avis clients afin d'aider une entreprise a mieux comprendre les attentes des consommateurs et a prendre des decisions operationnelles plus pertinentes.

**Auteurs**  
- Baha Sallami  
- Mariem Bouchaala

**Etablissement**  
ESPRIT

**Annee**  
2024-2025

---

## Remerciements

Nous adressons nos sinceres remerciements a notre encadrant pedagogique, Prof. Anis Bel Hadj Hassin, pour son accompagnement, ses recommandations et sa disponibilite tout au long de ce projet. Nous remercions egalement ESPRIT pour le cadre academique et technique offert, ainsi que toutes les personnes qui ont contribue, directement ou indirectement, a l'avancement de ce travail.

---

## Resume

Dans un environnement numerique ou les clients expriment en permanence leurs opinions sur les produits et services, l'analyse manuelle de milliers d'avis devient lente, couteuse et peu scalable. L'objectif de ce projet est de concevoir une application capable d'automatiser l'analyse des sentiments a partir d'avis clients, de detecter les principaux themes de satisfaction ou d'insatisfaction, et de transformer ces informations en indicateurs metiers exploitables.

L'application realisee repose sur une architecture modulaire en Python et sur une interface interactive developpee avec Streamlit. Elle prend en charge plusieurs modes d'acquisition des donnees: chargement de fichiers CSV, exploitation de jeux de donnees existants et integration d'un pipeline de scraping pour differentes plateformes. Une fois les donnees chargees, elles sont normalisees, nettoyees, dedupliquees et enrichies par plusieurs traitements NLP: analyse de sentiments avec VADER, comparaison optionnelle avec DistilBERT, extraction d'entites nommees avec spaCy, analyse par aspects, modelisation de sujets avec LDA et calcul d'indicateurs de performance orientes decision.

Le projet ne se limite pas a une demonstration algorithmique. Il propose un outil complet de pilotage qui permet de visualiser les tendances, les risques lies au sizing, les problemes de qualite, la perception du prix, les performances logistiques, ainsi que des recommandations automatiques basees sur des seuils analytiques. Les tests effectues sur les donnees du projet montrent que l'application est capable de produire une lecture utile du ressenti client et de convertir des donnees textuelles non structurees en information actionnable.

**Mots-cles :** NLP, analyse de sentiments, avis clients, Streamlit, VADER, DistilBERT, spaCy, LDA, aspect-based sentiment analysis, aide a la decision.

---

## Abstract

In digital commerce environments, customers continuously generate textual feedback about products, services, quality, delivery, sizing, and price perception. Manual analysis of these reviews is time-consuming and difficult to scale. The purpose of this project is to design and implement an application that automatically analyzes customer sentiment, highlights recurring positive and negative themes, and converts review data into business-oriented insights.

The developed solution is built in Python with a modular natural language processing pipeline and an interactive web interface using Streamlit. The system supports several data ingestion modes, including CSV uploads, existing datasets, and a scraping layer for selected sources. Once loaded, the data is normalized, cleaned, deduplicated, and enriched through multiple NLP stages: VADER sentiment analysis, optional DistilBERT comparison, named entity recognition with spaCy, aspect-based sentiment extraction, LDA topic modeling, and KPI generation.

Beyond sentiment classification, the project aims to support decision-making. The dashboard offers trend analysis, business indicators, entity exploration, aspect monitoring, topic discovery, and automated recommendations. Practical experimentation on the project dataset demonstrates that the application can successfully transform unstructured review text into interpretable analytical outputs that can support quality improvement and customer-experience optimization.

**Keywords:** sentiment analysis, customer reviews, NLP, dashboard, VADER, DistilBERT, topic modeling, NER, Streamlit, decision support.

---

## Table des matieres

1. Introduction generale  
2. Contexte et problematique  
3. Objectifs du projet  
4. Etude de l'existant et fondements theoriques  
5. Analyse des besoins et cahier des charges  
6. Conception et architecture de la solution  
7. Environnement technique et outils utilises  
8. Acquisition et preparation des donnees  
9. Mise en oeuvre des traitements NLP  
10. Realisation de l'application web  
11. Experimentation, validation et resultats  
12. Difficultes rencontrees et limites  
13. Perspectives d'amelioration  
14. Conclusion generale  
15. Bibliographie  
16. Annexes

---

## 1. Introduction generale

L'experience client est devenue un levier strategique majeur pour les entreprises. Dans le secteur de la vente en ligne, les avis consommateurs influencent non seulement la decision d'achat des futurs clients, mais aussi la perception de la marque, la fidelite et la reputation globale du service. Les entreprises disposent donc d'un volume important de donnees textuelles riches en information, mais difficiles a exploiter sans outils automatiques.

L'analyse automatique des sentiments constitue une reponse pertinente a ce besoin. Elle vise a detecter automatiquement l'orientation emotionnelle ou evaluative d'un texte, en distinguant par exemple les opinions positives, negatives ou neutres. Lorsqu'elle est combinee a une analyse par aspects, a l'extraction d'entites nommees et a des mecanismes de synthese, elle permet de depasser la simple classification pour fournir une lecture plus fine des causes de satisfaction ou d'insatisfaction.

Le present projet propose le developpement d'une application complete dediee a l'analyse d'avis clients. L'objectif ne se limite pas a appliquer un modele de sentiment sur un corpus. Il s'agit de construire une chaine de valeur complete: acquisition des donnees, normalisation, nettoyage, extraction d'informations, visualisation, mesure d'indicateurs metiers, et generation de recommandations decisionnelles.

Ce rapport presente l'ensemble de la demarche suivie, depuis la formulation du besoin jusqu'a la realisation finale de l'application. Il expose les choix techniques, l'architecture logicielle, les algorithmes retenus, les fonctionnalites mises en oeuvre, les resultats obtenus ainsi que les limites du systeme actuel.

---

## 2. Contexte et problematique

L'explosion du commerce electronique a transforme la relation entre les marques et leurs clients. Chaque fiche produit, chaque plateforme marchande et chaque service logistique produit aujourd'hui un volume important d'interactions numeriques. Parmi ces interactions, les avis clients occupent une place centrale. Ils constituent une source d'information precieuse pour les futurs acheteurs, mais aussi pour les entreprises qui souhaitent ameliorer leurs produits, corriger leurs dysfonctionnements et adapter leur communication.

Cependant, plusieurs problemes apparaissent rapidement.

D'abord, les avis sont non structures. Ils prennent la forme de phrases libres, parfois longues, parfois courtes, avec des fautes, des abreviations, des repetitions, des emojis, des doublons et des nuances. Ensuite, leur volume rend impossible une analyse strictement humaine a grande echelle. Enfin, une simple moyenne de notes n'est pas suffisante pour comprendre le pourquoi derriere l'opinion du client. Une note de trois etoiles ne dit pas si le probleme vient du tissu, de la taille, de la livraison ou du rapport qualite-prix.

Dans le domaine de la mode et du e-commerce textile, cette problematique est encore plus sensible. Les clients parlent souvent de qualite des matieres, de coupe, de taille, de longueur, de delai de livraison, de conformite visuelle, de confort ou encore de rapport qualite-prix. Une entreprise doit donc pouvoir identifier non seulement le sentiment global, mais aussi les dimensions precises qui posent probleme.

La problematique principale de ce projet peut donc etre formulee ainsi: **comment concevoir une application capable de transformer automatiquement un ensemble d'avis clients non structures en indicateurs analytiques et recommandations exploitables pour ameliorer l'experience client ?**

---

## 3. Objectifs du projet

Le projet poursuit plusieurs objectifs complementaires.

### 3.1 Objectif general

Concevoir et developper une application web interactive capable d'analyser automatiquement les sentiments exprimes dans des avis clients et de fournir une restitution claire, visuelle et exploitable pour l'aide a la decision.

### 3.2 Objectifs specifiques

Les objectifs specifiques du projet sont les suivants:

- permettre le chargement ou la collecte d'avis clients depuis plusieurs sources;
- normaliser les jeux de donnees heterogenes afin d'obtenir une structure exploitable;
- nettoyer les textes et eliminer les doublons ou contenus non exploitables;
- appliquer une analyse de sentiments rapide et robuste sur l'ensemble des avis;
- proposer une comparaison optionnelle avec un modele base sur Transformers;
- identifier les entites nommees presentes dans les commentaires;
- detecter les aspects les plus souvent mentionnes dans les avis;
- calculer des indicateurs metiers utiles a une entreprise;
- fournir une interface web moderne pour explorer les resultats;
- produire des recommandations automatiques basees sur des seuils de risque ou de satisfaction.

### 3.3 Valeur ajoutee du projet

La valeur ajoutee du projet reside dans son caractere integre. L'application ne se contente pas d'afficher un score de sentiment. Elle couvre l'ensemble du cycle analytique: ingestion, preparation, enrichissement, visualisation et interpretation decisionnelle. Elle peut donc etre envisagee comme un prototype de plateforme d'intelligence client oriente e-commerce.

---

## 4. Etude de l'existant et fondements theoriques

### 4.1 L'analyse de sentiments

L'analyse de sentiments, egalement appelee opinion mining, consiste a identifier la polarite emotionnelle d'un texte. Traditionnellement, trois classes sont utilisees: positif, negatif et neutre. Selon les besoins, cette analyse peut etre effectuee a plusieurs niveaux:

- au niveau document, lorsqu'on classe un avis entier;
- au niveau phrase, lorsqu'on classe chaque phrase separement;
- au niveau aspect, lorsqu'on relie l'opinion a une dimension precise du produit ou du service.

Les approches existantes peuvent etre regroupees en trois grandes familles:

- les approches lexicales, qui reposent sur des dictionnaires de mots positifs et negatifs;
- les approches supervises classiques, qui utilisent des representations vectorielles et des classifieurs appris;
- les approches neuronales profondes, notamment basees sur des modeles Transformers pre-entraines.

### 4.2 Approches lexicales: VADER

VADER est un analyseur lexical particulierement adapte aux textes courts et aux contenus issus du web. Il fournit plusieurs scores de polarite ainsi qu'un score compose global. Son principal avantage est sa rapidite d'execution et l'absence de phase d'entrainement. Dans ce projet, VADER a ete choisi comme moteur principal car il offre un excellent compromis entre simplicite, vitesse et lisibilite des resultats.

### 4.3 Modeles Transformers: DistilBERT

Les modeles Transformers tels que BERT ou DistilBERT permettent de capturer des relations contextuelles plus riches que les approches lexicales. Ils sont souvent plus performants sur des phrases ambigues ou formulees de maniere naturelle. En contrepartie, ils demandent plus de ressources et peuvent necessiter un telechargement de poids volumineux. Dans le projet, DistilBERT n'est pas utilise comme moteur principal, mais comme modele optionnel de comparaison afin de confronter les predictions d'un modele profond a celles de VADER.

### 4.4 Analyse de sentiments par aspects

Une analyse globale du sentiment n'est pas suffisante lorsqu'un avis melange des points positifs et negatifs. Par exemple, un client peut apprecier le design d'un produit tout en critiquant fortement sa taille. L'analyse par aspects permet de relier chaque opinion a une dimension specifique. Dans le cadre du projet, cinq aspects ont ete definis: **Quality**, **Sizing**, **Delivery**, **Price** et **Design**.

Cette approche est particulierement pertinente pour les produits vestimentaires, car elle met en evidence les causes operationnelles des insatisfactions: probleme de couture, taille trompeuse, prix juge eleve, delai de livraison excessif ou ecart visuel par rapport aux photos.

### 4.5 Extraction d'entites nommees

L'extraction d'entites nommees, ou NER, permet de detecter des noms de lieux, d'organisations, de marques, de produits ou d'autres expressions significatives dans un texte. Cette brique apporte une dimension qualitative supplementaire a l'analyse. Elle peut aider a identifier des marques citees, des lieux de livraison, des noms de gammes ou d'autres elements utiles a l'interpretation.

### 4.6 Topic Modeling

Le topic modeling vise a identifier automatiquement des themes recurrents dans un corpus. Dans ce projet, la methode LDA a ete retenue. Elle ne remplace pas l'analyse par aspects, mais la complete: alors que les aspects sont definis a l'avance, les sujets LDA emergent des donnees elles-memes. Cette combinaison permet de croiser lecture metier et decouverte exploratoire.

### 4.7 Justification des choix

Le projet adopte une approche hybride. VADER apporte la rapidite et la robustesse de base. DistilBERT apporte une comparaison contextuelle plus riche. L'analyse par aspects structure les retours clients selon des dimensions metier. Le NER apporte de la granularite. LDA revele des themes emergents. Ce choix architectural est pertinent dans un cadre academique car il permet de mobiliser plusieurs familles d'outils NLP au sein d'une application coherentement integree.

---

## 5. Analyse des besoins et cahier des charges

### 5.1 Besoins fonctionnels

Les besoins fonctionnels identifies sont les suivants:

1. Charger un jeu de donnees local au format CSV.
2. Accepter des variations de noms de colonnes lors de l'import.
3. Normaliser automatiquement les colonnes essentielles: texte, note, date, categorie, plateforme.
4. Nettoyer et preparer les textes avant analyse.
5. Analyser le sentiment global de chaque avis.
6. Comparer les predictions VADER et DistilBERT.
7. Extraire les entites nommees du texte.
8. Detecter les aspects mentionnes et le sentiment associe.
9. Decouvrir des sujets recurrents par topic modeling.
10. Afficher des tableaux de bord et graphiques interactifs.
11. Calculer des KPI metiers.
12. Generer des recommandations automatiques.
13. Permettre une analyse temps reel d'un texte saisi manuellement.
14. Exporter les resultats au format CSV.
15. Integrer un pipeline de scraping pour certaines sources.

### 5.2 Besoins non fonctionnels

Les besoins non fonctionnels sont egalement importants:

- interface claire et simple a manipuler;
- temps de reponse raisonnable sur des jeux de taille moderee;
- architecture modulaire facilitant la maintenance;
- robustesse face a des donnees incompletes ou heterogenes;
- capacite a fonctionner en environnement de salle de cours, y compris avec certaines dependances manquantes;
- possibilite de lancer facilement l'application sur machine locale.

### 5.3 Utilisateurs cibles

Deux profils d'utilisateurs peuvent etre envisages.

- Un profil academique ou analytique, qui souhaite etudier un corpus d'avis et comprendre les sorties des modeles NLP.
- Un profil metier ou decisionnel, interesse par les indicateurs, les tendances et les recommandations syntheses.

### 5.4 Scenario d'usage principal

Le scenario principal est le suivant:

1. L'utilisateur charge un fichier CSV ou recupere un jeu de donnees deja present.
2. L'application normalise la structure des colonnes.
3. Le pipeline de nettoyage et d'analyse est lance.
4. Les resultats sont stockes dans l'etat de session Streamlit.
5. L'utilisateur explore ensuite les pages du dashboard, des aspects, des sujets, des decisions, des tendances et du NER.
6. L'utilisateur exporte les resultats ou formule une interpretation metier.

---

## 6. Conception et architecture de la solution

### 6.1 Vue d'ensemble

La solution repose sur une architecture modulaire organisee en plusieurs couches:

- une couche d'acquisition des donnees;
- une couche de pretraitement;
- une couche de modelisation et d'enrichissement NLP;
- une couche de calcul metier;
- une couche de presentation et de visualisation.

Cette organisation permet de separer les responsabilites, d'ameliorer la lisibilite du projet et de faciliter l'evolution independante des composants.

### 6.2 Architecture des dossiers

La structure du projet est organisee autour des dossiers suivants:

- `app.py` comme point d'entree principal;
- `pages/` pour les pages Streamlit secondaires;
- `models/` pour les composants NLP;
- `preprocessing/` pour le nettoyage et la vectorisation;
- `utils/` pour les visualisations, les metriques et les KPI;
- `data/` pour les jeux de donnees et le chargement;
- `scraping/` pour la collecte d'avis a partir de plateformes externes;
- `assets/` pour le style visuel.

### 6.3 Architecture logique du pipeline

Le pipeline principal suit la sequence suivante:

1. lecture des donnees;
2. normalisation des colonnes;
3. nettoyage et deduplication;
4. tokenisation;
5. analyse de sentiments VADER;
6. extraction d'entites nommees;
7. vectorisation TF-IDF;
8. projection dense par TruncatedSVD;
9. modelisation des sujets LDA;
10. analyse par aspects;
11. calcul des KPI;
12. generation de recommandations;
13. visualisation et export.

### 6.4 Gestion de l'etat de l'application

Le projet utilise `st.session_state` pour conserver les objets de travail entre les pages de l'interface. Cela permet de charger une fois les donnees, d'executer l'analyse puis de naviguer entre les pages sans relancer la totalite du pipeline a chaque changement d'onglet.

### 6.5 Normalisation des colonnes d'entree

L'application ne se limite pas a un format CSV rigide. Une logique de mappage d'alias convertit automatiquement des variantes comme `review`, `comment`, `content` ou `review_text` vers la colonne cible `text`. De meme, `stars`, `score` ou `grade` sont rattaches a `rating`, et `created_at`, `timestamp` ou `posted_at` a `date`.

Ce choix est central dans le projet, car il renforce la robustesse de l'application face aux jeux de donnees reels provenant de Kaggle, de scrapers ou d'autres sources non homogenes.

---

## 7. Environnement technique et outils utilises

### 7.1 Langage et framework principal

Le projet est entierement developpe en Python. Ce langage est particulierement adapte aux projets de data science et de NLP grace a la richesse de son ecosysteme.

L'interface web a ete concue avec Streamlit. Ce framework offre un excellent compromis entre rapidite de developpement, lisibilite du code et ergonomie pour la visualisation de resultats analytiques.

### 7.2 Bibliotheques principales

Les principales dependances du projet sont les suivantes:

- `pandas` pour la manipulation tabulaire;
- `numpy` pour certaines operations numeriques;
- `scikit-learn` pour la vectorisation TF-IDF et la reduction de dimension;
- `nltk` et `vaderSentiment` pour l'analyse de sentiments lexicale;
- `spaCy` pour le NER et certaines operations linguistiques;
- `gensim` pour le LDA;
- `transformers` et `torch` pour DistilBERT;
- `plotly` pour les visualisations interactives;
- `selenium`, `requests`, `beautifulsoup4` et `webdriver-manager` pour la couche de scraping.

### 7.3 Choix de Streamlit

Le choix de Streamlit s'explique par plusieurs arguments:

- il permet de transformer rapidement un prototype analytique en application interactive;
- il s'integre naturellement avec pandas et Plotly;
- il facilite le decoupage en pages fonctionnelles;
- il est adapte a une demonstration academique ou a un POC metier.

### 7.4 Compatibilite et execution locale

L'application peut etre lancee localement via la commande suivante:

```bash
streamlit run app.py
```

Le projet inclut egalement des scripts de scraping dedies pouvant etre executes a part. Cette separation entre analyse et collecte rend l'application plus flexible.

---

## 8. Acquisition et preparation des donnees

### 8.1 Sources de donnees

Le projet exploite plusieurs types de sources:

- des jeux de donnees CSV deja presents dans le dossier `data/`;
- des jeux charges par l'utilisateur via l'interface;
- des donnees recuperees par des scrapers pour certaines plateformes;
- un fichier d'exemple servant de demonstration rapide.

Cette multiplicite des sources rend le systeme plus realiste, puisqu'en pratique une entreprise ne dispose pas toujours d'une base unique et proprement structuree.

### 8.2 Chargement et validation

Le module de chargement assure la lecture des fichiers CSV puis leur normalisation. Le systeme verifie ensuite la presence d'une colonne de texte. Si les colonnes `rating`, `date`, `category` ou `platform` sont absentes, elles peuvent etre completees automatiquement par des valeurs par defaut. Cela permet d'eviter les blocages inutiles tout en maintenant un format de travail coherent.

### 8.3 Nettoyage des donnees textuelles

Le nettoyage repose sur les operations suivantes:

- conversion des donnees en texte;
- suppression des URL;
- suppression des balises HTML;
- suppression des emojis;
- passage en minuscules;
- suppression de la ponctuation;
- normalisation des espaces;
- suppression des lignes vides;
- suppression des doublons apres normalisation du texte.

Cette phase est essentielle, car la qualite des resultats NLP depend fortement de la qualite des textes fournis au pipeline.

### 8.4 Tokenisation et preparation analytique

Apres nettoyage, les textes sont tokenises par expression reguliere. Les stopwords anglais sont retires. Une lemmatisation optionnelle est prevue lorsque les ressources spaCy sont disponibles. Les tokens sont ensuite concaténes pour produire une representation `token_text`, utilisee par les etapes de vectorisation et de topic modeling.

### 8.5 Observation sur le jeu `reviews.csv`

Une execution reelle du pipeline sur le fichier `data/reviews.csv` montre l'importance de la phase de nettoyage. Le fichier contient **500 lignes brutes**, mais apres normalisation, suppression des textes vides et deduplication, il ne reste que **32 avis exploitables et uniques**. Cette reduction signifie que le corpus initial contenait de nombreuses repetitions ou variations peu informatives. Cette observation constitue un resultat important du projet, car elle montre que l'etape de preparation n'est pas un detail technique, mais un facteur determinant de la qualite analytique.

---

## 9. Mise en oeuvre des traitements NLP

### 9.1 Analyse de sentiments avec VADER

Le module `sentiment_vader.py` constitue le coeur de l'analyse rapide. Pour chaque texte, il calcule les scores positifs, negatifs, neutres ainsi qu'un score compose `compound`. Ce score est ensuite converti en etiquette par les seuils suivants:

- `compound >= 0.05` -> Positive;
- `compound <= -0.05` -> Negative;
- sinon -> Neutral.

L'analyseur est applique a l'ensemble du DataFrame et les resultats sont ajoutes sous forme de nouvelles colonnes. Un calcul de distribution des labels est egalement disponible pour alimenter les graphiques et KPI.

### 9.2 Comparaison optionnelle avec DistilBERT

Le module `sentiment_bert.py` charge un modele `distilbert-base-uncased-finetuned-sst-2-english` via Hugging Face. Le choix de garder cette brique optionnelle est pertinent. En effet, le premier chargement du modele peut etre couteux en temps et en bande passante. L'application preserve donc une experience fluide avec VADER comme moteur par defaut, tout en proposant une comparaison plus contextuelle lorsque cela est utile.

Le systeme transforme egalement les scores de confiance faibles en sentiment neutre. Ce mecanisme est une adaptation pragmatique, car le modele de base est binaire alors que l'application souhaite conserver une classe `Neutral` pour les cas ambigus.

### 9.3 Extraction d'entites nommees

Le module `ner.py` essaie d'abord de charger `en_core_web_sm` de spaCy. Si ce modele n'est pas disponible, une strategie de repli basee sur des expressions regulieres est utilisee afin de garantir le fonctionnement du pipeline meme en environnement contraint.

Cette decision de conception est importante dans un projet academique. Elle montre une prise en compte des realites d'execution: l'application reste fonctionnelle meme lorsque toutes les ressources NLP ne sont pas installees. Chaque avis peut ainsi etre enrichi par:

- la liste des entites detectees;
- le nombre d'entites;
- un texte synthetique listant les entites.

### 9.4 Vectorisation TF-IDF et embeddings denses

Le module `vectorizer.py` transforme les textes tokenises en matrice TF-IDF. Les n-grammes de 1 a 2 termes sont pris en compte, ce qui permet de capturer a la fois des mots simples et certaines expressions courtes. Une reduction de dimension par TruncatedSVD est ensuite appliquee pour produire des embeddings denses de documents.

Cette approche presente deux avantages:

- elle fournit une representation plus compacte que la matrice creuse;
- elle evite de dependre exclusivement de modeles profonds pre-entraines.

Lors de l'execution sur `reviews.csv`, la matrice TF-IDF obtenue apres preparation a la taille **(32, 312)**, et les embeddings denses ont la taille **(32, 32)**.

### 9.5 Analyse de sentiments par aspects

Le module `aspect_analyzer.py` est specialement oriente vers l'analyse d'avis de produits textiles. Cinq aspects sont modelises:

- Quality;
- Sizing;
- Delivery;
- Price;
- Design.

Chaque aspect est associe a une liste de mots-cles metier. Lorsqu'un aspect est detecte dans un avis, le systeme recupere la phrase ou le contexte local correspondant et applique VADER sur cette portion de texte. Des regles d'override viennent ensuite corriger certains cas recurrentement mal interpretes par un score lexical pur. Par exemple, des expressions comme `runs small`, `too tight`, `overpriced` ou `see through` sont forcees vers une polarite negative lorsqu'elles apparaissent de maniere non ambiguë.

Cette strategie hybride entre lexique, contexte local et regles metier est particulierement adaptee a un projet applique. Elle permet d'obtenir une analyse plus interpretable que celle d'un simple classifieur global.

### 9.6 Topic Modeling par LDA

Le module `topic_model.py` applique une modelisation LDA sur les tokens prepares. Le nombre de sujets est parametre a 5 dans la configuration courante. LDA est utilise pour faire emerger des themes recurrents dans les commentaires. Chaque avis peut ensuite etre annote par son sujet dominant et un niveau de confiance.

Sur l'execution de reference, les sujets obtenus sont les suivants:

- Topic 1: Quality, Material, Dress
- Topic 2: Dress, Looks, Fabric
- Topic 3: Person, Suggest, Gorgeous
- Topic 4: Quality, Price, Dress
- Topic 5: Fits, Ordered, Flowy

Ces libelles confirment que le corpus traite se structure autour de la matiere, de l'apparence, de l'ajustement et de la perception de valeur.

### 9.7 Validation contre les notes

Le module `metrics.py` permet de convertir les notes en polarite de reference:

- 4 et 5 etoiles -> Positive;
- 3 etoiles -> Neutral;
- 1 et 2 etoiles -> Negative.

Cette transformation sert de base a une validation approximative du classifieur VADER. Sur l'execution de reference, la validation a donne les resultats suivants:

- Accuracy: **0.7812**
- F1 macro: **0.6960**
- Precision macro: **0.7261**
- Recall macro: **0.6952**

Ces valeurs sont encourageantes dans le cadre d'un prototype applique, meme si elles ne remplacent pas une evaluation sur un corpus manuellement annote.

---

## 10. Indicateurs metiers et moteur de decision

### 10.1 Objectif des KPI

L'un des apports majeurs de l'application est de traduire l'analyse textuelle en indicateurs interpretable par un decideur. Le module `kpi_calculator.py` calcule plusieurs KPI centraux pour le pilotage de l'experience client.

### 10.2 KPI implementes

Les principaux indicateurs sont:

- **Overall Customer Satisfaction**: moyenne ponderee des sentiments, avec Positive = 100, Neutral = 50 et Negative = 0.
- **Quality Score**: pourcentage de mentions positives liees a la qualite.
- **Return Risk Index**: pourcentage d'avis comportant des mentions negatives sur le sizing.
- **Delivery Score**: pourcentage de mentions positives sur la livraison.
- **Net Promoter Sentiment**: derive des notes 5 etoiles et 1 etoile.
- **Review Volume**: nombre total d'avis, avec repartition positive, neutre et negative.

Ces indicateurs font le lien entre NLP et prise de decision operationnelle.

### 10.3 Moteur de recommandation

Le module `decision_engine.py` applique des seuils explicites sur les taux de plainte ou de satisfaction. Par exemple:

- si les plaintes sur le sizing depassent 35 %, une alerte critique est generee;
- si la negativite sur la qualite depasse 30 %, une action corrective prioritaire est proposee;
- si les problemes de livraison deviennent importants, le systeme recommande de revoir la logistique;
- si certains sujets LDA connaissent une forte hausse recente, un avertissement est emis.

Ce mecanisme constitue une couche d'interpretation analytique. Il permet de passer d'une visualisation descriptive a une logique de decision assistee.

### 10.4 Interet pedagogique

Cette partie du projet est particulierement interessante d'un point de vue academique, car elle montre comment un resultat de NLP peut etre operationalise. L'etudiant ne s'arrete pas a la prediction de labels; il construit un outil d'aide a l'action.

---

## 11. Realisation de l'application web

### 11.1 Point d'entree principal

Le fichier `app.py` constitue le point d'entree principal de l'application. Il configure Streamlit, charge le style CSS, initialise l'etat de session et orchestre l'ensemble de l'analyse. Il contient egalement la logique de normalisation des donnees importees et la fonction `run_full_analysis`, qui pilote l'execution coordonnee des briques NLP.

### 11.2 Interface multipage

L'application est organisee en plusieurs pages Streamlit:

- **Dashboard**: vue synthetique des KPI, repartition des sentiments, comparaison par categorie et plateforme, apercu NER et table des avis.
- **Analyze**: analyse d'un avis unique, d'un mini-lot ou comparaison VADER vs DistilBERT.
- **Aspects**: exploration des aspects detectes et de leur polarite.
- **Topics**: visualisation des sujets LDA.
- **Decisions**: affichage des recommandations automatiques.
- **Trends**: suivi temporel du sentiment et des aspects.
- **NER**: exploration detaillee des entites nommees.

Ce decoupage apporte une bonne separation des usages. L'utilisateur peut naviguer selon une logique d'exploration progressive: vue globale, diagnostic, details, puis interpretation.

### 11.3 Analyse temps reel

La page `2_Analyze.py` permet d'analyser un texte saisi manuellement. Cette fonctionnalite est interessante pour la demonstration et pour des cas d'usage rapides. L'utilisateur peut tester un avis unique, analyser plusieurs lignes en batch ou comparer VADER et DistilBERT sur un meme texte.

### 11.4 Theme et ergonomie

L'application utilise une charte visuelle sombre coherente, renforcee par un fichier CSS dedie. Ce choix ameliore le contraste, la lisibilite des cartes analytiques et l'homogeneite visuelle globale. L'ergonomie du projet est un point important, car une solution analytique n'est utile que si ses sorties sont faciles a lire et a interpreter.

### 11.5 Export des resultats

Les resultats peuvent etre exportes au format CSV, ce qui permet de reutiliser les sorties du pipeline dans un autre contexte: analyse complementaire, rapport, partage ou archivage.

---

## 12. Couche de scraping et integration de sources externes

### 12.1 Objectif de la couche de scraping

Une application d'analyse de sentiments gagne fortement en valeur lorsqu'elle peut collecter les avis a la source. C'est pourquoi le projet inclut un dossier `scraping/` regroupant plusieurs scrapers et un pipeline d'orchestration.

### 12.2 Sources prises en charge

Le projet comprend des composants pour plusieurs plateformes ou cas d'usage:

- Yelp;
- Trustpilot;
- TripAdvisor;
- Amazon;
- SHEIN;
- des demonstrations ou sources experimentales supplementaires.

### 12.3 Normalisation des sorties de scraping

Le `ScraperPipeline` unifie les sorties en un format standard avec les colonnes suivantes:

- `text`
- `rating`
- `date`
- `platform`
- `category`

Ce choix s'aligne parfaitement avec le pipeline NLP principal, qui attend justement ces champs pour lancer l'analyse complete.

### 12.4 Limitations pratiques du scraping

Les plateformes e-commerce modernes protegent fortement leurs pages par des mecanismes anti-bot, des CAPTCHAs, des chargements JavaScript complexes et parfois des challenges de securite. Le projet a donc du adopter une strategie pragmatique:

- recours a Selenium et a des navigateurs reels pour certains cas;
- mode manuel pour permettre a l'utilisateur de resoudre un challenge avant la collecte;
- maintien d'un pipeline autonome base sur les CSV, afin que l'application principale reste independante du scraping en cas de blocage externe.

Cette experience montre qu'en contexte reel, la collecte de donnees est souvent la partie la plus fragile d'un systeme d'analyse d'avis. Le projet prend en compte cette realite et propose des solutions de contournement raisonnables.

---

## 13. Experimentation, validation et resultats

### 13.1 Protocole de test

Pour valider le comportement du systeme, plusieurs types de verification ont ete realises:

- compilation Python des fichiers principaux pour detecter d'eventuelles erreurs syntaxiques;
- execution de la pipeline NLP sur des jeux de donnees reels du projet;
- verification de la normalisation flexible des CSV;
- verification de l'exposition des pages et composants principaux;
- test de la comparaison VADER / DistilBERT;
- test de la presence ou non du modele spaCy pour le NER.

### 13.2 Resultats sur `data/reviews.csv`

Une execution complete du pipeline sur le fichier `data/reviews.csv` a fourni les observations suivantes:

- nombre de lignes brutes: **500**;
- nombre de lignes finales apres preparation: **32**;
- matrice TF-IDF: **(32, 312)**;
- embeddings denses: **(32, 32)**;
- modele spaCy NER disponible: **oui**;
- repartition VADER: **19 positifs**, **9 negatifs**, **4 neutres**.

Cette reduction du corpus a 32 lignes finales ne doit pas etre interpretee comme un echec. Elle montre au contraire que le corpus initial contenait beaucoup de bruit ou de duplications. Le pipeline nettoie donc fortement les entrees avant de produire les sorties analytiques.

### 13.3 Resultats de l'analyse par aspects

Sur cette meme execution, le resume des aspects detectes indique:

- **Quality**: 17 mentions, 8 positives, 7 negatives;
- **Design**: 14 mentions, 9 positives, 2 negatives;
- **Sizing**: 7 mentions, 6 positives, 1 negative;
- **Price**: 5 mentions, 1 positive, 3 negatives;
- **Delivery**: 4 mentions, 2 positives, 1 negative.

Ces chiffres sont riches d'interpretation. La qualite apparait comme le premier axe de discussion client, mais elle est aussi source de critiques non negligeables. Le design est plutot bien percu. Le prix, en revanche, semble plus sensible: peu de mentions, mais une proportion negative relativement forte. Le sizing est present, mais ici plutot favorable sur le corpus traite.

### 13.4 Resultats du topic modeling

Les cinq sujets LDA obtenus pointent principalement vers:

- la qualite et la matiere;
- l'apparence et l'esthetique du produit;
- la recommandation ou l'appreciation personnelle;
- la perception de valeur et de prix;
- l'ajustement ou la coupe.

Les sujets extraits sont coherents avec les aspects metier predefinis, ce qui renforce la credibilite globale du pipeline.

### 13.5 Resultats du NER

L'execution de reference a identifie **5 entites uniques** et **5 mentions d'entites**. Meme si ce nombre est limite sur ce corpus, il confirme que la brique NER est fonctionnelle. Dans un corpus plus riche en noms de marque, lieux, influenceurs ou points logistiques, cette fonctionnalite pourrait devenir plus strategique.

### 13.6 Validation par rapport aux notes

La comparaison entre le sentiment predit par VADER et le sentiment derive des notes donne:

- accuracy: 78.12 %;
- precision macro: 72.61 %;
- recall macro: 69.52 %;
- F1 macro: 69.60 %.

Ces scores montrent que le systeme fournit une lecture globalement pertinente de l'opinion du client. La classe neutre reste naturellement la plus delicate, ce qui est frequent dans les taches de sentiment ou les cas ambigus sont difficiles a trancher automatiquement.

### 13.7 Interpretation business des resultats

Au-dela des scores, l'application permet une lecture metier concrète:

- si les plaintes sur la qualite augmentent, l'entreprise doit verifier tissu, coutures et finition;
- si les problemes de prix se multiplient, il faut reevaluer le positionnement tarifaire ou la promesse de valeur;
- si la livraison degrade le ressenti global, l'entreprise peut agir sur les delais, la communication ou le partenaire logistique;
- si les retours lies au sizing augmentent, il faut revoir le guide des tailles ou enrichir les fiches produits.

Cette capacite a relier les sorties NLP a des actions concretes constitue l'un des points forts du projet.

---

## 14. Tests de robustesse et qualite de l'application

### 14.1 Verification syntaxique

Plusieurs composants essentiels ont ete verifies par compilation Python (`py_compile`), notamment l'application principale, les pages Streamlit et les scripts de scraping. Cette verification a permis de s'assurer que les modifications integrees restent executables.

### 14.2 Robustesse face aux fichiers CSV heterogenes

Un point important du projet a ete la gestion des jeux de donnees charges depuis des sources externes. Les essais ont montre qu'il etait insuffisant d'exiger strictement les colonnes `text`, `date`, `category` et `platform`. Une logique d'alias et de valeurs par defaut a donc ete introduite. Cette amelioration renforce nettement l'utilisabilite du systeme.

### 14.3 Robustesse face aux ressources NLP manquantes

Le projet gere plusieurs cas de ressources indisponibles:

- si le modele spaCy n'est pas present, le NER bascule sur une extraction regex;
- si DistilBERT n'est pas encore telecharge, l'application continue de fonctionner avec VADER;
- si certaines colonnes de donnees sont absentes, elles sont reconstruites ou completees automatiquement.

Cette philosophie de degradation elegante est un indicateur positif de maturite logicielle.

---

## 15. Difficultes rencontrees

### 15.1 Heterogeneite des donnees

Les jeux de donnees reels ne respectent pas toujours la meme convention de nommage. Cette heterogeneite a complique la premiere phase d'integration, d'ou la mise en place d'un mecanisme de normalisation des colonnes.

### 15.2 Presence de doublons

Le jeu `reviews.csv` contenait un volume important de redondance. Sans deduplication, les indicateurs auraient ete biaises. Cette difficulte confirme l'importance de ne pas considerer les donnees brutes comme directement exploitables.

### 15.3 Scraping bloque par les plateformes

Les plateformes comme Amazon ou SHEIN peuvent activer des protections anti-bot. Cela a impose l'utilisation de navigateurs controles, de sessions manuelles et de strategies plus prudentes. Ce point est moins un probleme de code qu'une contrainte structurelle de l'environnement web moderne.

### 15.4 Equilibre entre precision et simplicite

Le choix de VADER comme moteur principal privilegie la rapidite et la simplicite. Un modele profond pourrait parfois offrir une meilleure finesse contextuelle, mais au prix d'une execution plus lourde. Le projet a donc cherche un equilibre raisonnable entre performance, accesibilite et robustesse.

---

## 16. Limites du systeme

Malgre ses points forts, le projet presente plusieurs limites.

### 16.1 Limites liees aux donnees

- la qualite du resultat depend fortement de la qualite des avis en entree;
- les corpus trop petits ou trop repetitifs limitent la pertinence des sujets LDA;
- la validation par les notes n'est qu'une approximation de la verite terrain.

### 16.2 Limites liees aux modeles

- VADER peut mal traiter l'ironie, le sarcasme ou certaines formulations complexes;
- DistilBERT utilise ici un modele generaliste et non un modele finement adapte au domaine textile;
- l'analyse par aspects reste basee sur des mots-cles definis manuellement, ce qui peut laisser passer certains synonymes ou formulations implicites.

### 16.3 Limites liees au scraping

- les protections anti-bot rendent certaines collectes instables;
- certains sites changent regulierement leur structure HTML;
- la collecte multi-plateforme complete peut necessiter une maintenance reguliere.

### 16.4 Limites liees a l'evaluation

Le projet n'integre pas encore un corpus de verite terrain annote manuellement par experts sur lequel comparer plusieurs modeles de facon rigoureuse. Une telle evaluation constituerait une extension naturelle pour un travail futur.

---

## 17. Perspectives d'amelioration

Plusieurs pistes d'evolution peuvent etre envisagees.

### 17.1 Ameliorations algorithmiques

- fine-tuning d'un modele BERT sur un corpus d'avis e-commerce;
- ajout d'une classification multiclasse plus fine, par exemple tres positif, positif, neutre, negatif, tres negatif;
- integration d'une analyse multilingue pour traiter des avis en francais, arabe ou d'autres langues;
- enrichissement des dictionnaires d'aspects avec apprentissage semi-automatique.

### 17.2 Ameliorations metier

- suivi par produit, par collection ou par fournisseur;
- comparaison de performances entre plateformes;
- alertes automatiques en temps reel selon l'evolution des KPI;
- integration de recommandations plus explicites pour les equipes qualite, marketing et logistique.

### 17.3 Ameliorations techniques

- persistance en base de donnees au lieu d'un traitement uniquement en memoire;
- authentification utilisateur et gestion de profils;
- export PDF ou generation automatique de rapports;
- planification de collectes automatiques selon un calendrier;
- instrumentation plus poussee pour le suivi des erreurs et de la performance.

### 17.4 Ameliorations UX

- ajout de captures comparatives avant/apres analyse;
- integration d'un assistant textuel de synthese;
- personnalisation des vues pour un profil data analyst ou un profil manager;
- generation de rapports executive summary en un clic.

---

## 18. Conclusion generale

Ce projet a permis de concevoir et de realiser une application complete d'analyse automatique des sentiments orientee experience client. Au lieu de se limiter a un simple exercice d'algorithmique, la solution proposee couvre l'ensemble de la chaine de valeur analytique: collecte ou chargement des avis, normalisation, nettoyage, enrichissement NLP, calcul d'indicateurs, visualisation interactive et recommandation metier.

D'un point de vue technique, le projet mobilise plusieurs briques complementaires: VADER pour une analyse rapide et interpretable, DistilBERT pour la comparaison, spaCy pour le NER, TF-IDF et TruncatedSVD pour les representations vectorielles, LDA pour la decouverte de sujets, ainsi qu'un moteur d'analyse par aspects adapte au domaine textile. Ce choix hybride est pertinent car il allie performance pratique, modularite et pedagogie.

D'un point de vue fonctionnel, l'application apporte une reelle valeur en rendant lisibles des avis clients bruts et volumineux. Elle aide a identifier ce qui plait, ce qui deçoit et ce qui doit etre corrige en priorite. Elle rapproche ainsi le traitement automatique du langage de besoins concrets lies a la qualite produit, au sizing, a la livraison et a la perception du prix.

Enfin, ce travail met en evidence une realite essentielle des projets data appliques: la valeur ne provient pas uniquement du modele, mais de la qualite du pipeline global. La normalisation des donnees, la robustesse logicielle, l'ergonomie de restitution et l'orientation decisionnelle sont tout aussi importantes que la precision d'un classifieur. Le projet constitue donc une base solide pour des extensions futures et un excellent support academique pour illustrer la mise en pratique du NLP dans un cas d'usage metier reel.

---

## 19. Bibliographie

1. Hutto, C. J., and Gilbert, E. E. VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Proceedings of ICWSM, 2014.
2. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.
3. Sanh, V., Debut, L., Chaumond, J., and Wolf, T. DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. NeurIPS Workshop, 2019.
4. Rehurek, R., and Sojka, P. Software Framework for Topic Modelling with Large Corpora. Proceedings of LREC Workshop, 2010.
5. Explosion AI. Documentation officielle de spaCy.
6. Hugging Face. Documentation officielle Transformers.
7. Streamlit. Documentation officielle Streamlit.
8. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.
9. McKinney, W. Data Structures for Statistical Computing in Python. Proceedings of the Python in Science Conference, 2010.
10. Plotly Technologies Inc. Documentation officielle Plotly.

---

## 20. Annexes

### Annexe A - Commandes utiles

Lancement de l'application principale:

```bash
streamlit run app.py
```

Lancement du scraper Amazon en mode manuel:

```bash
python run_scraper.py --manual --product "<URL_ou_ASIN>" --name "Nom du produit" --pages 2
```

Lancement du scraper SHEIN en mode manuel:

```bash
python run_shein_scraper.py --manual
```

### Annexe B - Colonnes standard du pipeline

Le pipeline principal travaille autour des colonnes suivantes:

- `text`
- `rating`
- `date`
- `category`
- `platform`
- `review_id`

### Annexe C - Suggestions de mise en page pour atteindre 25 pages academiques

Pour obtenir une version finale d'environ 25 pages dans Word ou PDF, il est recommande de:

- inserer une page de garde institutionnelle complete avec logo;
- ajouter un sommaire automatique;
- inserer des captures d'ecran des pages Streamlit;
- ajouter un diagramme d'architecture et un diagramme de pipeline;
- utiliser une police 12 avec interligne 1.5 et marges standard;
- inserer les tableaux de resultats et les figures generees par l'application.

### Annexe D - Captures d'ecran a inserer

Les captures d'ecran les plus pertinentes pour la version finale du rapport sont:

1. page d'accueil et zone de chargement des donnees;
2. dashboard KPI;
3. page d'analyse temps reel;
4. page d'analyse par aspects;
5. page des sujets LDA;
6. page des decisions automatiques;
7. page NER;
8. exemple d'export CSV.

---

## 21. Synthese finale a reutiliser lors de la soutenance

En quelques phrases, ce projet propose une application web intelligente capable de lire des avis clients, de comprendre leur polarite, d'identifier les themes problematiques et de produire des recommandations utiles pour ameliorer l'experience client. Sa force repose sur l'integration coherente de plusieurs techniques NLP dans une interface simple a exploiter. Il ne s'agit pas seulement d'un classifieur de sentiments, mais d'un veritable tableau de bord de pilotage de la voix du client.