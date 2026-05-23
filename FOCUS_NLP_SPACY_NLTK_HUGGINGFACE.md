# Focus NLP - spaCy, NLTK, Hugging Face

Ce document est une fiche ciblee pour la soutenance sur les briques NLP suivantes :

- spaCy
- NLTK
- Hugging Face
- BERT et DistilBERT

Important : dans le code actuel du projet, **le modele effectivement utilise via Hugging Face est DistilBERT**, pas un BERT complet fine-tune separement. Il faut donc bien distinguer **la famille BERT** du **modele concret implemente dans l'application**.

---

## 1. spaCy dans notre projet

### Question
Pourquoi avons-nous utilise spaCy dans ce projet ?

### Reponse
spaCy a ete utilise pour deux besoins principaux :

- l'extraction d'entites nommees dans le module NER ;
- la segmentation en phrases dans l'analyse par aspects.

Nous l'avons choisi parce qu'il offre une pipeline NLP prete a l'emploi, rapide et robuste pour des taches linguistiques structurees. Dans notre projet, spaCy n'est pas utilise pour tout le pipeline, mais pour des taches ciblées ou sa valeur est forte.

### Code
- [models/ner.py](models/ner.py#L25)
- [models/ner.py](models/ner.py#L35)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L229)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L234)

---

## 2. Comment spaCy est-il utilise pour le NER ?

### Question
Comment fonctionne la partie NER basee sur spaCy ?

### Reponse
Le module charge d'abord `en_core_web_sm`. Si le modele est disponible, il applique `doc.ents` sur chaque texte pour extraire les entites, leur label et leurs positions. Les resultats sont ensuite injectes dans le DataFrame avec le nombre d'entites et une version textuelle resumee.

### Code
- [models/ner.py](models/ner.py#L31)
- [models/ner.py](models/ner.py#L43)
- [models/ner.py](models/ner.py#L62)
- [models/ner.py](models/ner.py#L81)

---

## 3. Que se passe-t-il si spaCy ou son modele ne sont pas disponibles ?

### Question
Pourquoi le projet ne depend-il pas strictement de spaCy pour fonctionner ?

### Reponse
Nous avons prevu un fallback pour que l'application reste utilisable en environnement local ou de salle de TP. Si `en_core_web_sm` n'est pas disponible, le NER passe sur une extraction regex conservatrice. Pour la segmentation des phrases dans l'analyse par aspects, on peut aussi basculer sur un `spacy.blank("en")` avec `sentencizer`, ou sur un split regex si besoin.

Ce point montre que notre pipeline est robuste et qu'il degrade proprement au lieu d'echouer.

### Code
- [models/ner.py](models/ner.py#L35)
- [models/ner.py](models/ner.py#L111)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L231)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L236)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L237)

---

## 4. Pourquoi spaCy n'est-il pas utilise pour tout le projet ?

### Question
Pourquoi ne pas avoir tout fait avec spaCy ?

### Reponse
Parce que nous avons adopte une approche pragmatique. spaCy est tres bon pour la structure linguistique, le NER et certaines transformations, mais il n'est pas le meilleur choix pour tous les besoins du projet. Pour le sentiment global, VADER via NLTK est plus leger et plus rapide. Pour la comparaison deep learning, Hugging Face est plus pertinent. Nous avons donc choisi le meilleur outil selon la tache.

### Code
- [models/ner.py](models/ner.py#L14)
- [models/sentiment_vader.py](models/sentiment_vader.py#L47)
- [models/sentiment_bert.py](models/sentiment_bert.py#L15)

---

## 5. NLTK dans notre projet

### Question
Quel est le role de NLTK dans notre application ?

### Reponse
NLTK intervient a deux niveaux principaux :

- la gestion des stopwords pendant le preprocessing ;
- le support du lexique VADER si la bibliotheque `vaderSentiment` n'est pas disponible.

Nous utilisons donc NLTK comme une base lexicale legere et pratique, surtout pour le nettoyage et le sentiment rule-based.

### Code
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L17)
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L23)
- [models/sentiment_vader.py](models/sentiment_vader.py#L16)
- [models/sentiment_vader.py](models/sentiment_vader.py#L21)

---

## 6. Comment NLTK est-il utilise dans le preprocessing ?

### Question
Que fait exactement NLTK pendant le nettoyage des textes ?

### Reponse
Dans `TextCleaner`, NLTK sert a recuperer les stopwords anglais. Lors de la tokenisation, les tokens juges trop frequents et peu informatifs sont retires. Cela permet de conserver des mots plus utiles pour la vectorisation, les aspects et le topic modeling.

Si les ressources NLTK ne sont pas presentes, nous avons aussi un petit ensemble de stopwords de secours defini en dur.

### Code
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L19)
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L24)
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L26)
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L69)

---

## 7. Comment NLTK est-il lie a VADER ?

### Question
Pourquoi NLTK apparait-il aussi dans le module de sentiment ?

### Reponse
Le module de sentiment essaie d'abord d'utiliser `vaderSentiment.vaderSentiment`. Si cette dependance n'est pas disponible, il bascule sur `nltk.sentiment.vader`. Dans ce cas, il verifie aussi la presence du lexique `vader_lexicon` et le telecharge si necessaire.

Cette logique rend l'analyse de sentiments plus portable entre environnements.

### Code
- [models/sentiment_vader.py](models/sentiment_vader.py#L13)
- [models/sentiment_vader.py](models/sentiment_vader.py#L17)
- [models/sentiment_vader.py](models/sentiment_vader.py#L21)
- [models/sentiment_vader.py](models/sentiment_vader.py#L23)

---

## 8. Pourquoi avoir choisi une approche rule-based avec VADER ?

### Question
Pourquoi ne pas avoir utilise directement un modele deep learning partout ?

### Reponse
Parce que l'objectif etait d'avoir une base fiable, rapide et demonstrable en local. VADER permet une analyse quasi instantanee, sans phase d'entrainement, avec des scores tres lisibles. Pour une interface Streamlit et des demonstrations temps reel, c'est un choix tres pratique.

Le deep learning reste present dans le projet, mais comme couche complementaire et non comme dependance obligatoire.

### Code
- [models/sentiment_vader.py](models/sentiment_vader.py#L47)
- [pages/2_Analyze.py](pages/2_Analyze.py#L62)
- [models/sentiment_bert.py](models/sentiment_bert.py#L15)

---

## 9. Hugging Face dans notre projet

### Question
Quel est le role de Hugging Face dans l'application ?

### Reponse
Hugging Face est utilise pour charger un pipeline de sentiment base sur DistilBERT. Cette brique sert de modele de comparaison avancee face a VADER. Elle n'est pas obligatoire pour le fonctionnement de base, mais elle permet de montrer l'integration d'un modele Transformer moderne dans le projet.

### Code
- [models/sentiment_bert.py](models/sentiment_bert.py#L15)
- [models/sentiment_bert.py](models/sentiment_bert.py#L46)
- [models/sentiment_bert.py](models/sentiment_bert.py#L48)

---

## 10. BERT vs DistilBERT

### Question
Quelle est la difference entre BERT et DistilBERT, et lequel utilisons-nous ?

### Reponse
BERT est un modele Transformer bidirectionnel de reference, tres puissant mais relativement lourd. DistilBERT est une version compressee de BERT : plus legere, plus rapide et moins couteuse en ressources, avec une perte de performance souvent acceptable.

Dans notre projet, nous utilisons **DistilBERT**, pas un BERT complet distinct. Il faut donc dire en soutenance que nous exploitons la famille BERT via une version distillee adaptee a un usage applicatif local.

### Code
- [models/sentiment_bert.py](models/sentiment_bert.py#L16)
- [models/sentiment_bert.py](models/sentiment_bert.py#L18)

---

## 11. Quel modele Hugging Face utilisons-nous exactement ?

### Question
Quel est le modele pre-entraine charge dans le projet ?

### Reponse
Le modele charge est `distilbert-base-uncased-finetuned-sst-2-english`. C'est un modele Hugging Face pre-entraine pour la classification de sentiment en anglais. Il est bien adapte a une demonstration rapide, meme s'il reste generaliste et non fine-tune sur des avis e-commerce textiles.

### Code
- [models/sentiment_bert.py](models/sentiment_bert.py#L18)
- [models/sentiment_bert.py](models/sentiment_bert.py#L48)

---

## 12. Pourquoi DistilBERT n'est-il pas le modele principal ?

### Question
Pourquoi avoir garde DistilBERT comme modele optionnel ?

### Reponse
Parce qu'il est plus lourd que VADER et demande un chargement du pipeline Hugging Face. Pour une application Streamlit locale, nous voulions que le projet reste rapide et utilisable meme sans GPU ou sans telechargement prealable du modele. DistilBERT a donc ete integre comme comparaison de haut niveau, pas comme moteur obligatoire.

### Code
- [models/sentiment_bert.py](models/sentiment_bert.py#L2)
- [models/sentiment_bert.py](models/sentiment_bert.py#L33)
- [models/sentiment_bert.py](models/sentiment_bert.py#L46)
- [app.py](app.py#L252)

---

## 13. Comment Hugging Face est-il instancie dans le code ?

### Question
Comment le modele est-il charge techniquement ?

### Reponse
Le chargement se fait via `transformers.pipeline("sentiment-analysis", ...)`. Le modele est charge a la demande dans la methode `_load`, ce qui evite de penaliser le demarrage de l'application quand on n'utilise pas DistilBERT. C'est une forme de lazy loading.

### Code
- [models/sentiment_bert.py](models/sentiment_bert.py#L42)
- [models/sentiment_bert.py](models/sentiment_bert.py#L46)
- [models/sentiment_bert.py](models/sentiment_bert.py#L48)

---

## 14. Pourquoi avez-vous ajoute la classe Neutral alors que SST-2 est binaire ?

### Question
Comment gerer la neutralite avec DistilBERT alors que le modele est binaire ?

### Reponse
Le modele renvoie surtout positif ou negatif. Pour l'adapter a notre application, nous avons defini une regle simple : si la confiance est trop faible, nous reclassons la prediction en `Neutral`. Cela permet d'avoir une sortie plus coherente avec le reste de l'application, qui distingue positif, negatif et neutre.

### Code
- [models/sentiment_bert.py](models/sentiment_bert.py#L56)
- [models/sentiment_bert.py](models/sentiment_bert.py#L61)

---

## 15. Comment comparez-vous VADER et DistilBERT ?

### Question
Comment montrez-vous la difference entre approche lexicale et Transformer ?

### Reponse
Nous proposons une page d'analyse temps reel ou un meme texte peut etre passe a VADER puis a DistilBERT. L'application affiche ensuite les deux resultats et indique si les modeles sont d'accord ou non. C'est utile pour illustrer les differences entre une approche rule-based rapide et une approche contextuelle basee sur un Transformer.

### Code
- [pages/2_Analyze.py](pages/2_Analyze.py#L157)
- [models/sentiment_vader.py](models/sentiment_vader.py#L55)
- [models/sentiment_bert.py](models/sentiment_bert.py#L67)

---

## 16. Quelle reponse donner si le professeur demande : "Pourquoi ces trois bibliotheques ensemble ?"

### Reponse orale conseillee
Nous avons combine spaCy, NLTK et Hugging Face parce qu'elles repondent a des besoins differents et complementaires. spaCy est tres utile pour le NER et la structure linguistique, NLTK apporte des ressources lexicales et un sentiment rule-based leger via VADER, et Hugging Face nous permet d'integrer un modele Transformer moderne avec DistilBERT. Au lieu de tout faire avec une seule bibliotheque, nous avons choisi l'outil le plus adapte a chaque tache.

### Code
- [models/ner.py](models/ner.py#L35)
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L23)
- [models/sentiment_vader.py](models/sentiment_vader.py#L13)
- [models/sentiment_bert.py](models/sentiment_bert.py#L46)

---

## 17. Quelle reponse donner si le professeur demande : "Pourquoi ne pas avoir fine-tune BERT ?"

### Reponse orale conseillee
Parce que l'objectif du projet etait de livrer une application complete, modulaire et executable localement, pas seulement d'optimiser un benchmark de classification. Un fine-tuning BERT metier serait une excellente extension, mais il demande un corpus annote, du temps d'entrainement et davantage de ressources. Nous avons donc privilegie une solution hybride plus realiste pour un prototype academique demonstrable.

### Code
- [models/sentiment_bert.py](models/sentiment_bert.py#L18)
- [models/sentiment_vader.py](models/sentiment_vader.py#L47)
- [app.py](app.py#L238)

---

## 18. Reponse courte a memoriser

### Version tres courte
Dans notre projet, spaCy sert surtout au NER et a la segmentation en phrases, NLTK sert au preprocessing lexical et au support de VADER, et Hugging Face sert a integrer DistilBERT comme modele Transformer de comparaison. Nous n'avons pas utilise un BERT full fine-tune specifique au domaine, mais une version distillee plus legere et plus pratique pour une application locale.
