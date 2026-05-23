# DOCUMENTATION COMPLETE DU PROJET - PHASE PAR PHASE

## PROJET 1 | Analyse Automatique des Sentiments pour Ameliorer l'Experience Client

**Projet realise par :** Baha Sallami, Mariem Bouchaala  
**Etablissement :** ESPRIT  
**Annee universitaire :** 2024-2025

---

## 1. Presentation generale du projet

Ce projet consiste a developper une application web intelligente capable de lire des avis clients, de les nettoyer, de les analyser automatiquement avec plusieurs techniques NLP, puis de transformer ces avis en informations utiles pour l'aide a la decision.

Le projet ne se limite pas a une classification simple des sentiments. Il couvre tout le cycle complet de traitement :

1. acquisition ou chargement des donnees ;
2. normalisation et nettoyage ;
3. analyse de sentiments ;
4. extraction d'entites ;
5. analyse par aspects ;
6. topic modeling ;
7. calcul d'indicateurs metier ;
8. generation de recommandations ;
9. visualisation dans une interface Streamlit.

L'objectif principal est d'aider une entreprise a mieux comprendre l'experience client a partir d'avis textuels non structures.

---

## 2. Vision globale du systeme

L'application repose sur une architecture modulaire. Chaque partie du projet remplit un role bien defini.

### 2.1 Organisation des dossiers

- `app.py` : point d'entree principal de l'application.
- `pages/` : pages Streamlit secondaires.
- `models/` : composants NLP et logique analytique.
- `preprocessing/` : nettoyage et vectorisation.
- `utils/` : KPI, metriques et visualisations.
- `data/` : fichiers de donnees et chargement.
- `scraping/` : collecte d'avis depuis des plateformes externes.
- `assets/` : style visuel.

### 2.2 Flux global

Le flux logique du projet est le suivant :

1. l'utilisateur charge un CSV ou utilise un dataset existant ;
2. les colonnes sont normalisees ;
3. le texte est nettoye et deduplique ;
4. les sentiments sont analyses ;
5. les entites et les aspects sont extraits ;
6. les sujets recurrents sont detectes ;
7. les KPI sont calcules ;
8. des recommandations sont generees ;
9. l'ensemble est affiche dans le dashboard.

---

## 3. Phase 1 - Acquisition des donnees

### 3.1 Objectif de cette phase

La premiere phase consiste a recuperer les avis clients. Ces avis peuvent venir de plusieurs sources.

### 3.2 Sources prises en charge

Le projet accepte :

- des fichiers CSV importes par l'utilisateur ;
- des datasets deja presents dans `data/` ;
- des donnees issues de scrapers ;
- des jeux d'exemple utilises pour la demonstration.

### 3.3 Pourquoi cette phase est importante

Dans un projet reel, les donnees ne sont pas toujours propres ni homogenes. Cette phase permet donc de recuperer les avis de maniere flexible avant de lancer les traitements NLP.

### 3.4 Modules utilises

- `data/data_loader.py`
- `app.py`
- `scraping/scraper_pipeline.py`

### 3.5 Fonctions principales

- chargement des fichiers CSV ;
- lecture des datasets locaux ;
- normalisation des colonnes ;
- ajout de valeurs par defaut si certaines colonnes manquent.

### 3.6 Colonnes standard attendues

Le systeme travaille principalement avec les colonnes suivantes :

- `text`
- `rating`
- `date`
- `category`
- `platform`
- `review_id`

---

## 4. Phase 2 - Normalisation des donnees

### 4.1 Pourquoi normaliser les donnees

Les fichiers reels n'utilisent pas toujours les memes noms de colonnes. Un dataset peut contenir `review`, un autre `comment`, un autre `content`. Sans normalisation, il faudrait adapter manuellement chaque import.

### 4.2 Comment cela fonctionne

Le projet utilise un systeme d'alias dans `app.py`. Plusieurs variantes de noms de colonnes sont mappees vers une structure unique.

Exemples :

- `review`, `review_text`, `comment`, `content` -> `text`
- `stars`, `score`, `grade` -> `rating`
- `created_at`, `timestamp`, `posted_at` -> `date`
- `source`, `site`, `website` -> `platform`

### 4.3 Que fait le systeme si une colonne manque

Si certaines colonnes sont absentes :

- `rating` devient vide ;
- `date` peut etre remplacee par la date du jour ;
- `category` est remplacee par `General` ;
- `platform` devient `Uploaded CSV`.

### 4.4 Interet technique

Cette phase rend le projet robuste face aux datasets heterogenes, notamment ceux venant de Kaggle ou des scrapers externes.

---

## 5. Phase 3 - Nettoyage des donnees textuelles

### 5.1 Objectif

Avant de lancer les modeles NLP, il faut convertir les avis en un format propre et exploitable.

### 5.2 Operations de nettoyage

Le module `preprocessing/cleaner.py` applique les etapes suivantes :

1. conversion du contenu en texte ;
2. suppression des URL ;
3. suppression des balises HTML ;
4. suppression des emojis ;
5. passage en minuscules ;
6. suppression de la ponctuation ;
7. normalisation des espaces ;
8. suppression des lignes vides ;
9. suppression des doublons sur le texte nettoye.

### 5.3 Tokenisation

Apres nettoyage, le texte est tokenise avec une expression reguliere. Seuls les tokens utiles sont conserves.

### 5.4 Suppression des stopwords

Le projet utilise NLTK pour retirer les stopwords anglais, c'est-a-dire les mots tres frequents mais peu informatifs comme `the`, `is`, `and`, `of`, `to`.

### 5.5 Lemmatisation

Une lemmatisation optionnelle est possible via spaCy si les ressources sont disponibles. Cela permet de ramener des mots differents a leur forme canonique.

### 5.6 Pourquoi cette phase est cruciale

Sans nettoyage, les modeles travailleraient sur du bruit, des repetitions et des donnees peu pertinentes. Cette phase conditionne fortement la qualite des resultats.

---

## 6. Phase 4 - Analyse de sentiments principale avec VADER

### 6.1 Pourquoi VADER

VADER est le moteur principal de sentiment dans le projet. Il a ete choisi parce qu'il est :

- rapide ;
- interpretable ;
- leger ;
- adapte aux textes courts et aux avis web.

### 6.2 Ce que fait VADER

Pour chaque avis, VADER calcule :

- un score `positive` ;
- un score `negative` ;
- un score `neutral` ;
- un score global `compound`.

Ensuite, le score `compound` est converti en etiquette de sentiment :

- `compound >= 0.05` -> `Positive`
- `compound <= -0.05` -> `Negative`
- sinon -> `Neutral`

### 6.3 Comment VADER est utilise dans le projet

VADER est applique automatiquement a tous les avis dans `NLPPipeline`. Il ajoute les colonnes suivantes dans le DataFrame :

- `vader_label`
- `vader_compound`
- `vader_positive`
- `vader_negative`
- `vader_neutral`
- `vader_confidence`

### 6.4 Pourquoi ne pas avoir utilise seulement un modele deep learning

Parce qu'un modele deep learning est plus lourd, plus lent et plus couteux a charger. VADER offre une excellente base rapide pour une application interactive locale.

---

## 7. Phase 5 - Comparaison avec DistilBERT via Hugging Face

### 7.1 Positionnement de cette phase

DistilBERT n'est pas le moteur principal. Il est utilise comme modele optionnel de comparaison.

### 7.2 BERT vs DistilBERT

- **BERT** est un modele Transformer bidirectionnel de reference.
- **DistilBERT** est une version compressee de BERT, plus legere et plus rapide.

Dans le projet, le modele charge est :

`distilbert-base-uncased-finetuned-sst-2-english`

### 7.3 Pourquoi DistilBERT a ete choisi

Il permet :

- d'introduire une brique deep learning moderne ;
- de comparer une approche lexicale a une approche contextuelle ;
- de garder un cout de calcul raisonnable.

### 7.4 Comment Hugging Face est utilise

Le projet utilise `transformers.pipeline("sentiment-analysis", ...)` pour charger le modele a la demande.

### 7.5 Pourquoi le chargement est optionnel

Le premier chargement du modele peut etre long. Si on l'activait systematiquement, cela ralentirait toute l'application. C'est pourquoi DistilBERT est charge seulement si l'utilisateur demande la comparaison.

### 7.6 Gestion de la classe neutre

Le modele SST-2 est essentiellement binaire. Pour rester coherent avec l'application, une prediction de faible confiance est reclassifiee en `Neutral`.

### 7.7 Resultat attendu de cette phase

L'objectif n'est pas de remplacer VADER, mais de comparer deux philosophies :

- une approche lexicale rapide ;
- une approche Transformer contextuelle.

---

## 8. Phase 6 - Extraction d'entites nommees avec spaCy

### 8.1 Objectif

Cette phase sert a identifier des mots ou groupes de mots significatifs, par exemple des noms de marque, de produit, de lieu ou d'organisation.

### 8.2 Pourquoi spaCy

spaCy est utilise pour le NER car il propose une pipeline linguistique tres efficace et directement exploitable.

### 8.3 Fonctionnement

Le projet essaie d'abord de charger `en_core_web_sm`. Si le modele est disponible, il applique `doc.ents` sur le texte.

Pour chaque entite, il stocke :

- le texte de l'entite ;
- le label de l'entite ;
- la position de debut ;
- la position de fin.

### 8.4 Fallback si spaCy n'est pas disponible

Si le modele n'est pas installe, le projet utilise une extraction regex plus simple. Cela permet de garder un pipeline fonctionnel meme en environnement contraint.

### 8.5 Sorties produites

Le DataFrame est enrichi par exemple avec :

- `ner_entities`
- `ner_entity_count`
- `ner_entity_text`

Ensuite, un resume des entites les plus frequentes peut etre genere.

---

## 9. Phase 7 - Vectorisation et representations numeriques

### 9.1 Objectif

Transformer les textes nettoyes en representations numeriques exploitables.

### 9.2 TF-IDF

Le projet utilise `TfidfVectorizer` de scikit-learn. Cette etape produit une matrice creuse basee sur les mots et les n-grammes.

### 9.3 Pourquoi TF-IDF

TF-IDF est :

- simple a comprendre ;
- efficace sur des corpus textuels classiques ;
- interpretable ;
- rapide a calculer.

### 9.4 Embeddings denses via TruncatedSVD

Une fois la matrice TF-IDF calculee, le projet applique `TruncatedSVD` afin d'obtenir une representation dense plus compacte.

### 9.5 Pourquoi faire cela

Cette phase permet de compresser l'information, de faciliter certaines analyses et d'eviter de dependre uniquement de modeles pre-entranes lourds.

### 9.6 Exemple de resultat observe

Sur le dataset de reference, la matrice TF-IDF obtenue etait de taille `(32, 312)` et les embeddings denses de taille `(32, 32)`.

---

## 10. Phase 8 - Analyse de sentiments par aspects

### 10.1 Pourquoi cette phase

Un avis peut contenir plusieurs opinions sur plusieurs dimensions du produit. Il faut donc depasser le simple sentiment global.

### 10.2 Aspects definis dans le projet

Les aspects retenus sont :

- `Quality`
- `Sizing`
- `Delivery`
- `Price`
- `Design`

### 10.3 Comment les aspects sont detectes

Chaque aspect est associe a une liste de mots-cles metier. Si ces mots apparaissent dans un avis, l'aspect est considere comme mentionne.

### 10.4 Comment le sentiment d'aspect est calcule

Le systeme recupere les phrases ou le contexte local contenant les mots-cles, puis applique VADER sur cette portion du texte.

### 10.5 Regles d'override metier

Certaines expressions comme `runs small`, `too tight`, `overpriced` ou `see through` ont un sens metier fort. Le projet ajoute donc des regles deterministes pour corriger la polarite quand cela est pertinent.

### 10.6 Sorties de cette phase

Cette phase produit :

- un DataFrame enrichi avec les colonnes d'aspects ;
- une table longue de mentions d'aspects ;
- un resume par aspect ;
- une evolution mensuelle par aspect ;
- une direction de tendance.

---

## 11. Phase 9 - Topic Modeling avec LDA

### 11.1 Objectif

Identifier automatiquement les themes recurrents dans les avis sans definir tous les sujets a l'avance.

### 11.2 Methode utilisee

Le projet utilise LDA via `gensim`.

### 11.3 Comment cela fonctionne

1. les textes sont preprocessees ;
2. une representation sac de mots est construite ;
3. LDA apprend plusieurs topics ;
4. chaque avis recoit un topic dominant ;
5. des labels lisibles sont generes a partir des mots dominants.

### 11.4 Pourquoi LDA est utile ici

Les aspects sont predefinis, mais les topics permettent de decouvrir des themes emergents. Les deux approches sont complementaires.

### 11.5 Resultat concret observe

Les topics obtenus sur le dataset de reference portaient notamment sur :

- qualite et matiere ;
- apparence et style ;
- recommendation et appreciation ;
- prix et valeur ;
- fit et coupe.

---

## 12. Phase 10 - Validation des predictions

### 12.1 Objectif

Verifier si la classification de sentiment est globalement coherente.

### 12.2 Methode choisie

Le projet convertit les notes clients en classes de sentiment :

- 4-5 etoiles -> `Positive`
- 3 etoiles -> `Neutral`
- 1-2 etoiles -> `Negative`

Ensuite, ces classes sont comparees aux predictions VADER.

### 12.3 Metriques calculees

- accuracy ;
- precision ;
- recall ;
- F1 macro ;
- matrice de confusion.

### 12.4 Limite de cette methode

Ce n'est pas une verite terrain annotee manuellement. C'est une validation approximative, mais utile pour un prototype applique.

### 12.5 Resultats observes

Sur le dataset de reference, les resultats obtenus etaient d'environ :

- accuracy : 78.12 % ;
- precision macro : 72.61 % ;
- recall macro : 69.52 % ;
- F1 macro : 69.60 %.

---

## 13. Phase 11 - Calcul des KPI metier

### 13.1 But de cette phase

Transformer les resultats NLP en indicateurs directement lisibles par un decideur.

### 13.2 KPI implementes

- `Overall Customer Satisfaction`
- `Quality Score`
- `Return Risk Index`
- `Delivery Score`
- `Net Promoter Sentiment`
- `Review Volume`

### 13.3 Comment ces KPI sont calcules

- certains sont bases sur la distribution des sentiments ;
- d'autres sur les aspects positifs ou negatifs ;
- d'autres sur les notes 5 etoiles et 1 etoile.

### 13.4 Pourquoi cette phase est importante

Elle fait le lien entre traitement du langage et besoin metier. Le projet devient un outil d'aide a la decision et pas seulement un exercice NLP.

---

## 14. Phase 12 - Generation automatique de recommandations

### 14.1 Objectif

Produire des actions concretes a partir des resultats analytiques.

### 14.2 Principe

Le module `decision_engine.py` applique des seuils sur les KPI, les aspects et les evolutions des topics.

### 14.3 Exemples de recommandations

- si les plaintes de sizing sont trop fortes, revoir le guide des tailles ;
- si la qualite est trop critiquee, auditer la production ;
- si la livraison est problematique, revoir la logistique ;
- si un topic augmente soudainement, generer une alerte.

### 14.4 Valeur ajoutee

Cette phase transforme l'analyse descriptive en decision assistee.

---

## 15. Phase 13 - Interface utilisateur avec Streamlit

### 15.1 Pourquoi Streamlit

Streamlit a ete choisi car il permet de construire rapidement une application analytique interactive en Python.

### 15.2 Role de `app.py`

`app.py` :

- configure la page ;
- charge le CSS ;
- initialise `st.session_state` ;
- charge les donnees ;
- lance le pipeline ;
- sauvegarde les resultats.

### 15.3 Pages de l'application

- **Dashboard** : vue globale et KPI ;
- **Analyze** : analyse d'un avis ou comparaison VADER/DistilBERT ;
- **Aspects** : analyse par aspects ;
- **Topics** : sujets LDA ;
- **Decisions** : recommandations automatiques ;
- **Trends** : tendances temporelles ;
- **NER** : exploration des entites.

### 15.4 Role de `st.session_state`

Il permet de partager les resultats entre les pages sans recalculer tout le pipeline.

### 15.5 Export

Les resultats peuvent etre exportes au format CSV.

---

## 16. Phase 14 - Scraping et collecte externe

### 16.1 Pourquoi cette phase existe

Pour qu'une application d'analyse de sentiments soit utile, elle doit pouvoir recuperer des avis a la source.

### 16.2 Plateformes prises en charge

Le projet inclut des scrapers pour :

- Yelp ;
- Trustpilot ;
- TripAdvisor ;
- Amazon ;
- SHEIN.

### 16.3 Pourquoi le scraping est separe du coeur analytique

Le scraping depend fortement de plateformes externes, de leur HTML et de leurs protections. Il est donc plus fragile. Le separer de l'application analytique permet de garder un coeur applicatif stable.

### 16.4 Cas d'Amazon

Le scraper Amazon peut extraire un ASIN et ouvrir une session manuelle si un CAPTCHA apparait.

### 16.5 Cas de SHEIN

SHEIN peut afficher un challenge de securite. Le projet ouvre alors un navigateur et laisse l'utilisateur rendre la page accessible avant de lancer l'extraction des reviews.

### 16.6 Pipeline unifie

Le `ScraperPipeline` harmonise les sorties des differentes sources en un format unique compatible avec le pipeline NLP.

---

## 17. Phase 15 - Robustesse et gestion des cas limites

### 17.1 Robustesse face aux fichiers incomplets

Si un fichier CSV n'a pas toutes les colonnes attendues, le systeme essaie de les reconstruire ou de les remplacer par des valeurs par defaut.

### 17.2 Robustesse face aux ressources NLP manquantes

- spaCy indisponible -> fallback regex pour le NER ;
- DistilBERT indisponible -> l'application continue avec VADER ;
- ressources NLTK absentes -> listes minimales de secours si necessaire.

### 17.3 Robustesse face aux scrapers bloques

Le projet propose des modes manuels lorsque les plateformes activent des protections anti-bot.

### 17.4 Verification technique

Plusieurs fichiers du projet ont ete verifies via `py_compile` pour s'assurer de leur validite syntaxique.

---

## 18. Phase 16 - Resultats observes sur le dataset de reference

### 18.1 Corpus

- 500 lignes brutes chargees ;
- 32 lignes finales exploitables apres nettoyage.

### 18.2 Sentiment global

- 19 avis positifs ;
- 9 avis negatifs ;
- 4 avis neutres.

### 18.3 Aspects les plus visibles

- Quality : 17 mentions ;
- Design : 14 mentions ;
- Sizing : 7 mentions ;
- Price : 5 mentions ;
- Delivery : 4 mentions.

### 18.4 NER

- 5 entites uniques detectees ;
- 5 mentions totales.

### 18.5 Validation

- accuracy : 78.12 % ;
- F1 macro : 69.60 %.

### 18.6 Interpretation

Le projet montre que les axes dominants du corpus sont la qualite, le design, le prix et la coupe. Il met aussi en evidence les risques operationnels possibles pour une entreprise.

---

## 19. Phase 17 - Limites du projet

### 19.1 Limites liees aux modeles

- VADER peut etre limite sur l'ironie et le sarcasme ;
- DistilBERT reste generaliste et non fine-tune sur le textile ;
- les aspects reposent sur des mots-cles metier definis a la main.

### 19.2 Limites liees aux donnees

- les datasets reels peuvent etre tres bruites ;
- les petits corpus limitent la stabilite des topics ;
- la validation par les notes reste approximative.

### 19.3 Limites liees au scraping

- protections anti-bot ;
- HTML instable ;
- maintenance reguliere necessaire.

---

## 20. Phase 18 - Perspectives d'amelioration

### 20.1 Cote NLP

- fine-tuning d'un modele BERT metier ;
- support multilingue ;
- enrichissement automatique des aspects ;
- classification plus fine.

### 20.2 Cote applicatif

- base de donnees ;
- authentification ;
- export PDF ;
- alertes automatiques ;
- rapport executive summary automatique.

### 20.3 Cote metier

- suivi par produit ;
- suivi par collection ;
- comparaison multi-plateformes ;
- recommandations specialisees selon l'equipe cible.

---

## 21. Resume final du projet

Ce projet est une application NLP complete qui transforme des avis clients en indicateurs metier et recommandations actionnables.

Il combine :

- nettoyage et preprocessing ;
- analyse de sentiments avec VADER ;
- comparaison avec DistilBERT ;
- extraction d'entites via spaCy ;
- analyse par aspects ;
- topic modeling par LDA ;
- calcul de KPI ;
- generation de recommandations ;
- visualisation interactive avec Streamlit.

Sa force principale est son approche de bout en bout : acquisition, nettoyage, enrichissement, visualisation et aide a la decision.

---

## 22. Fichiers les plus importants du projet

### Application principale

- `app.py`

### Pages Streamlit

- `pages/1_Dashboard.py`
- `pages/2_Analyze.py`
- `pages/2_Aspects.py`
- `pages/3_Topics.py`
- `pages/4_Decisions.py`
- `pages/5_Trends.py`
- `pages/6_NER.py`

### Modules NLP

- `models/nlp_pipeline.py`
- `models/sentiment_vader.py`
- `models/sentiment_bert.py`
- `models/ner.py`
- `models/aspect_analyzer.py`
- `models/topic_model.py`
- `models/decision_engine.py`

### Preprocessing

- `preprocessing/cleaner.py`
- `preprocessing/vectorizer.py`

### Outils metier

- `utils/kpi_calculator.py`
- `utils/metrics.py`
- `utils/visualizations.py`

### Donnees et scraping

- `data/data_loader.py`
- `scraping/scraper_pipeline.py`
- `scraping/amazon_scraper.py`
- `scraping/shein_scraper.py`

---

## 23. Conclusion documentaire

Ce document sert de reference complete phase par phase pour comprendre toute l'architecture du projet. Il peut etre utilise comme documentation technique, comme support de soutenance ou comme base pour une future evolution du systeme.