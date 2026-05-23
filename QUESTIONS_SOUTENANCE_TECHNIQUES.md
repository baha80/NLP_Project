# Questions Techniques de Soutenance

Ce document regroupe les questions techniques les plus probables pendant la soutenance, avec des reponses detaillees et les emplacements correspondants dans le code source.

---

## 1. Architecture generale du projet

### Question
Pourquoi avez-vous choisi une architecture modulaire et multipage ?

### Reponse
Nous avons choisi une architecture modulaire pour separer clairement les responsabilites. Le fichier principal ne contient pas toute la logique applicative. Il orchestre le flux global, tandis que les modules specialises gerent chacun une tache precise : preprocessing, analyse de sentiment, NER, aspects, topic modeling, calcul des KPI, recommandations et scraping.

Cette approche presente plusieurs avantages :

- meilleure maintenabilite ;
- code plus lisible ;
- possibilite de tester ou d'ameliorer un composant sans casser tout le systeme ;
- reutilisation plus simple des modules dans d'autres contextes.

Le choix multipage avec Streamlit permet de separer les usages fonctionnels. L'utilisateur ne voit pas une seule page surchargee, mais plusieurs pages specialisees : dashboard, analyse temps reel, aspects, topics, decisions, tendances et NER.

### Emplacements dans le code
- [app.py](app.py#L97)
- [app.py](app.py#L110)
- [models/nlp_pipeline.py](models/nlp_pipeline.py#L19)
- [pages/1_Dashboard.py](pages/1_Dashboard.py#L32)
- [pages/2_Analyze.py](pages/2_Analyze.py#L57)
- [pages/6_NER.py](pages/6_NER.py#L41)

---

## 2. Pipeline complet de traitement

### Question
Quel est le pipeline complet de l'application, de l'import des donnees jusqu'aux recommandations ?

### Reponse
Le pipeline global suit une chaine logique bien definie.

1. Chargement des donnees.
2. Normalisation des noms de colonnes.
3. Validation des colonnes obligatoires.
4. Nettoyage et deduplication des avis.
5. Tokenisation des textes.
6. Analyse de sentiment avec VADER.
7. Extraction d'entites nommees.
8. Vectorisation TF-IDF.
9. Generation d'embeddings denses via TruncatedSVD.
10. Comparaison optionnelle avec DistilBERT.
11. Analyse par aspects.
12. Topic modeling avec LDA.
13. Calcul des KPI.
14. Generation de recommandations metier.
15. Sauvegarde des resultats dans la session Streamlit pour affichage sur les pages.

La fonction d'orchestration principale est `run_full_analysis`, alors que `NLPPipeline` gere la chaine NLP de base.

### Emplacements dans le code
- [app.py](app.py#L238)
- [models/nlp_pipeline.py](models/nlp_pipeline.py#L39)

---

## 3. Gestion des CSV heterogenes

### Question
Comment avez-vous gere les datasets provenant de Kaggle ou d'autres sources avec des colonnes differentes ?

### Reponse
Nous avons introduit une logique de normalisation flexible des colonnes. Au lieu d'imposer strictement des noms comme `text`, `rating`, `date`, `category` et `platform`, nous avons defini des alias. Par exemple, `review`, `comment`, `content` ou `review_text` peuvent etre reconnus comme du texte. De la meme maniere, `stars`, `score` et `grade` peuvent etre convertis en `rating`.

Si certaines colonnes sont absentes, l'application complete les valeurs avec des valeurs par defaut. Par exemple, si la date manque, elle peut etre remplacee par la date du jour. Si la plateforme manque, la valeur `Uploaded CSV` est utilisee.

Cela rend l'application beaucoup plus robuste face aux datasets reels.

### Emplacements dans le code
- [app.py](app.py#L34)
- [app.py](app.py#L186)
- [data/data_loader.py](data/data_loader.py#L31)
- [data/data_loader.py](data/data_loader.py#L49)

---

## 4. Pretraitement des textes

### Question
Que fait exactement la phase de preprocessing ?

### Reponse
La phase de preprocessing nettoie et prepare les avis avant l'analyse. Elle effectue les operations suivantes :

- suppression des URL ;
- suppression des balises HTML ;
- suppression des emojis ;
- passage en minuscules ;
- suppression de la ponctuation ;
- normalisation des espaces ;
- tokenisation par expression reguliere ;
- suppression des stopwords ;
- lemmatisation optionnelle si spaCy est disponible ;
- suppression des avis vides ou dupliques.

Cette etape est indispensable car un modele NLP travaille mieux sur des textes propres et homogenes.

### Emplacements dans le code
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L53)
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L63)
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L89)

---

## 5. Forte reduction du corpus apres nettoyage

### Question
Pourquoi le nombre d'avis diminue-t-il fortement apres pretraitement ?

### Reponse
Parce que nous eliminons tout ce qui nuit a la qualite analytique : lignes vides, textes non exploitables, contenus trop faibles et doublons. Sur notre execution reelle du projet, le fichier `reviews.csv` contenait 500 lignes brutes, mais seulement 32 avis exploitables et uniques apres nettoyage et deduplication.

Ce n'est pas un probleme. Au contraire, cela montre que la phase de preparation est efficace. Sans cela, les analyses seraient faussees par des repetitions ou du bruit.

### Emplacements dans le code
- [preprocessing/cleaner.py](preprocessing/cleaner.py#L89)
- [models/nlp_pipeline.py](models/nlp_pipeline.py#L39)

---

## 6. Choix de VADER comme moteur principal

### Question
Pourquoi avez-vous choisi VADER comme modele principal de sentiment ?

### Reponse
VADER a ete choisi comme moteur principal pour plusieurs raisons.

- Il est rapide.
- Il ne necessite pas d'entrainement.
- Il est bien adapte aux textes courts et aux avis web.
- Il fournit des scores interpretable : positif, negatif, neutre et compound.

Dans une application interactive Streamlit, il est important d'avoir une analyse fluide. VADER apporte ce compromis entre simplicite, rapidite et lisibilite.

### Emplacements dans le code
- [models/sentiment_vader.py](models/sentiment_vader.py#L47)
- [models/sentiment_vader.py](models/sentiment_vader.py#L55)
- [models/sentiment_vader.py](models/sentiment_vader.py#L80)

---

## 7. Role de DistilBERT

### Question
Pourquoi DistilBERT est-il optionnel et non le modele principal ?

### Reponse
DistilBERT est plus riche contextuellement, mais il est aussi plus lourd. Il demande plus de ressources, plus de temps de chargement et depend du telechargement du modele. Nous l'avons donc garde comme modele de comparaison ou d'analyse avancee, sans en faire une dependance obligatoire du flux principal.

Ce choix garantit une bonne experience utilisateur tout en montrant que le projet sait integrer un modele Transformer moderne.

### Emplacements dans le code
- [models/sentiment_bert.py](models/sentiment_bert.py#L15)
- [models/sentiment_bert.py](models/sentiment_bert.py#L67)
- [models/sentiment_bert.py](models/sentiment_bert.py#L72)
- [app.py](app.py#L252)

---

## 8. Gestion de la classe neutre avec DistilBERT

### Question
Comment avez-vous gere le fait que le modele DistilBERT utilise soit naturellement binaire ?

### Reponse
Le modele choisi est base sur SST-2, donc il produit surtout positif ou negatif. Pour rester coherent avec notre application qui exploite aussi la classe `Neutral`, nous avons introduit un seuil de confiance. Quand le score du modele est trop faible, la prediction est reclassifiee comme neutre.

Cela permet d'eviter une polarite artificielle sur les avis ambigus.

### Emplacements dans le code
- [models/sentiment_bert.py](models/sentiment_bert.py#L23)
- [models/sentiment_bert.py](models/sentiment_bert.py#L56)

---

## 9. Extraction d'entites nommees

### Question
Comment fonctionne votre NER et que se passe-t-il si spaCy n'est pas installe ?

### Reponse
Le module NER essaie d'abord de charger le modele `en_core_web_sm` de spaCy. Si ce modele est disponible, on utilise une vraie extraction d'entites nommees. Sinon, l'application ne s'arrete pas. Elle bascule vers une methode de secours basee sur des expressions regulieres, capable d'extraire des suites de mots capitalises qui ressemblent a des noms de marque, de lieu ou de produit.

Cette logique de fallback garantit la robustesse du projet en environnement local ou partiellement configure.

### Emplacements dans le code
- [models/ner.py](models/ner.py#L14)
- [models/ner.py](models/ner.py#L43)
- [models/ner.py](models/ner.py#L62)
- [models/ner.py](models/ner.py#L81)
- [models/ner.py](models/ner.py#L111)

---

## 10. TF-IDF et embeddings denses

### Question
Pourquoi avoir utilise TF-IDF puis TruncatedSVD ?

### Reponse
TF-IDF est une representation simple, robuste et interpretable du texte. Elle permet de capturer l'importance relative des mots dans le corpus. Ensuite, TruncatedSVD est applique pour produire des vecteurs denses plus compacts, ce qui facilite certaines analyses sans recourir systematiquement a un modele profond.

Cette approche est adaptee a un projet academique local, car elle reste efficace tout en etant legere.

### Emplacements dans le code
- [preprocessing/vectorizer.py](preprocessing/vectorizer.py#L50)
- [preprocessing/vectorizer.py](preprocessing/vectorizer.py#L77)
- [preprocessing/vectorizer.py](preprocessing/vectorizer.py#L121)

---

## 11. Analyse par aspects

### Question
Pourquoi avez-vous implemente une analyse par aspects en plus du sentiment global ?

### Reponse
Le sentiment global dit si un avis est positif ou negatif, mais il ne dit pas pourquoi. Un client peut aimer le design et detester la taille dans le meme commentaire. L'analyse par aspects permet de lier l'opinion a une dimension metier precise.

Nous avons defini cinq aspects adaptes au domaine textile et e-commerce :

- Quality ;
- Sizing ;
- Delivery ;
- Price ;
- Design.

Chaque aspect est detecte par mots-cles, puis le sentiment est evalue localement sur la phrase ou le contexte pertinent.

### Emplacements dans le code
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L249)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L260)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L320)

---

## 12. Regles metier dans l'analyse par aspects

### Question
Pourquoi avez-vous ajoute des regles d'override dans l'analyse par aspects ?

### Reponse
Certaines expressions tres metier ne sont pas toujours bien capturees par un score lexical general. Par exemple, `runs small`, `too tight`, `overpriced` ou `see through` sont des indices tres forts d'insatisfaction dans un contexte textile. Nous avons donc ajoute des regles deterministes qui corrigent la polarite dans certains cas evidents.

Cela ameliore l'interpretabilite et la pertinence des resultats au niveau metier.

### Emplacements dans le code
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L448)

---

## 13. Topic modeling avec LDA

### Question
Pourquoi avoir ajoute du topic modeling alors que les aspects existent deja ?

### Reponse
Les aspects sont definis a l'avance selon une logique metier. Le topic modeling, au contraire, sert a faire emerger automatiquement des themes recurrents a partir des donnees. Les deux approches sont donc complementaires.

- Les aspects repondent a des questions que l'on connait deja.
- Les topics permettent de decouvrir des themes non prevus explicitement.

Cette double approche enrichit l'analyse.

### Emplacements dans le code
- [models/topic_model.py](models/topic_model.py#L15)
- [models/topic_model.py](models/topic_model.py#L64)
- [models/topic_model.py](models/topic_model.py#L111)
- [models/topic_model.py](models/topic_model.py#L145)
- [models/topic_model.py](models/topic_model.py#L167)

---

## 14. Validation du modele par les notes

### Question
Comment avez-vous evalue votre analyse de sentiment ?

### Reponse
Nous avons effectue une validation indirecte a partir des notes clients. Les notes ont ete converties en etiquettes de sentiment :

- 4 et 5 etoiles vers `Positive` ;
- 3 etoiles vers `Neutral` ;
- 1 et 2 etoiles vers `Negative`.

Ensuite, nous avons compare ces etiquettes avec les predictions VADER pour calculer l'accuracy, la precision, le recall, le F1 macro et la matrice de confusion.

Ce n'est pas une annotation manuelle experte, mais c'est une methode pertinente pour un prototype academique.

### Emplacements dans le code
- [utils/metrics.py](utils/metrics.py#L17)
- [utils/metrics.py](utils/metrics.py#L140)
- [utils/metrics.py](utils/metrics.py#L163)

---

## 15. KPI metier

### Question
Quels KPI calcule votre application et quelle est leur utilite ?

### Reponse
L'application calcule plusieurs KPI metier :

- Overall Customer Satisfaction ;
- Quality Score ;
- Return Risk Index ;
- Delivery Score ;
- Net Promoter Sentiment ;
- Review Volume.

Ces indicateurs permettent de transformer des avis textuels en informations directement exploitables par un responsable qualite, produit ou logistique.

### Emplacements dans le code
- [utils/kpi_calculator.py](utils/kpi_calculator.py#L14)
- [pages/1_Dashboard.py](pages/1_Dashboard.py#L35)

---

## 16. Moteur de decision

### Question
Comment votre application passe-t-elle d'une analyse descriptive a des recommandations concrètes ?

### Reponse
Le moteur de decision applique des seuils metier sur les aspects et les tendances. Par exemple :

- si les plaintes de sizing depassent un seuil, le systeme recommande de revoir le guide des tailles ;
- si la qualite est trop critiquee, il propose un audit de production ;
- si la livraison degrade les avis, il recommande une action sur la logistique ;
- si un topic augmente brutalement, il genere une alerte.

Chaque recommandation est classee par priorite.

### Emplacements dans le code
- [models/decision_engine.py](models/decision_engine.py#L12)
- [models/decision_engine.py](models/decision_engine.py#L17)
- [models/decision_engine.py](models/decision_engine.py#L153)
- [pages/4_Decisions.py](pages/4_Decisions.py#L24)

---

## 17. Utilisation de st.session_state

### Question
Comment les differentes pages de l'application partagent-elles les resultats ?

### Reponse
Nous utilisons `st.session_state` pour stocker l'etat global de l'analyse. Une fois que le pipeline est lance, les resultats sont enregistres dans la session. Les autres pages peuvent alors les reutiliser directement sans recalculer tout le pipeline.

Cette approche ameliore les performances et rend l'experience utilisateur plus fluide.

### Emplacements dans le code
- [app.py](app.py#L110)
- [app.py](app.py#L135)
- [app.py](app.py#L316)
- [pages/1_Dashboard.py](pages/1_Dashboard.py#L32)
- [pages/6_NER.py](pages/6_NER.py#L39)

---

## 18. Analyse temps reel

### Question
Que permet exactement la page d'analyse temps reel ?

### Reponse
La page d'analyse temps reel permet trois usages :

- analyser un seul avis avec score et entites extraites ;
- analyser plusieurs avis saisis manuellement ;
- comparer VADER et DistilBERT sur un meme texte.

Cette page est utile pedagogiquement car elle montre le comportement du systeme en direct, sans passer par un dataset complet.

### Emplacements dans le code
- [pages/2_Analyze.py](pages/2_Analyze.py#L29)
- [pages/2_Analyze.py](pages/2_Analyze.py#L57)
- [pages/2_Analyze.py](pages/2_Analyze.py#L60)
- [pages/2_Analyze.py](pages/2_Analyze.py#L157)

---

## 19. Dashboard principal

### Question
Que montre le dashboard principal ?

### Reponse
Le dashboard principal montre les KPI, les distributions de sentiments, les comparaisons par categorie et plateforme, un apercu NER et une table filtrable des avis analyses. Il sert de point d'entree global pour comprendre rapidement l'etat du corpus et les indicateurs essentiels.

### Emplacements dans le code
- [pages/1_Dashboard.py](pages/1_Dashboard.py#L32)
- [pages/1_Dashboard.py](pages/1_Dashboard.py#L71)
- [pages/1_Dashboard.py](pages/1_Dashboard.py#L95)

---

## 20. Page NER

### Question
Que permet la page NER dans l'application ?

### Reponse
La page NER permet d'explorer les entites extraites : nombre d'entites uniques, volume de mentions, tableau resume, graphique des entites les plus frequentes et filtrage des avis contenant certaines entites. Elle donne une lecture qualitative supplementaire du corpus.

### Emplacements dans le code
- [pages/6_NER.py](pages/6_NER.py#L15)
- [pages/6_NER.py](pages/6_NER.py#L41)
- [pages/6_NER.py](pages/6_NER.py#L57)
- [pages/6_NER.py](pages/6_NER.py#L104)

---

## 21. Pipeline de scraping unifie

### Question
Pourquoi avez-vous cree un ScraperPipeline au lieu d'appeler les scrapers directement ?

### Reponse
Le `ScraperPipeline` sert a unifier les sorties de plusieurs sources et a produire un format commun compatible avec la pipeline NLP. Il gere la normalisation des colonnes, le nettoyage minimal, la fusion des DataFrames, la suppression des doublons et l'export final.

Cela facilite l'integration multi-source et decouple le scraping de l'analyse.

### Emplacements dans le code
- [scraping/scraper_pipeline.py](scraping/scraper_pipeline.py#L28)
- [scraping/scraper_pipeline.py](scraping/scraper_pipeline.py#L108)
- [scraping/scraper_pipeline.py](scraping/scraper_pipeline.py#L141)

---

## 22. Scraper Amazon

### Question
Comment votre scraper Amazon fonctionne-t-il techniquement ?

### Reponse
Le scraper Amazon extrait d'abord l'ASIN a partir d'une URL produit ou d'une reference. Ensuite, il peut ouvrir un navigateur en mode manuel pour laisser l'utilisateur resoudre un CAPTCHA ou une authentification. Une fois la page accessible, il charge les pages d'avis Amazon et parse les cartes de reviews avec Selenium.

Cette approche est necessaire car Amazon protege fortement ses pages.

### Emplacements dans le code
- [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L21)
- [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L126)
- [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L293)
- [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L301)
- [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L305)
- [run_scraper.py](run_scraper.py#L12)

---

## 23. Scraper SHEIN

### Question
Pourquoi avez-vous implemente un mode manuel pour SHEIN ?

### Reponse
SHEIN active souvent un challenge de securite ou un controle anti-bot, ce qui rend les requetes directes peu fiables. Nous avons donc implemente un scraper base sur un vrai navigateur, avec detection du challenge et reprise manuelle. L'utilisateur rend la page utilisable, puis le scraper analyse le DOM deja charge pour extraire les reviews, notes, dates et identifiants de produit.

Cette solution est pragmatique et adaptee aux contraintes reelles du web moderne.

### Emplacements dans le code
- [scraping/shein_scraper.py](scraping/shein_scraper.py#L25)
- [scraping/shein_scraper.py](scraping/shein_scraper.py#L130)
- [scraping/shein_scraper.py](scraping/shein_scraper.py#L155)
- [scraping/shein_scraper.py](scraping/shein_scraper.py#L436)
- [scraping/shein_scraper.py](scraping/shein_scraper.py#L450)
- [run_shein_scraper.py](run_shein_scraper.py#L17)

---

## 24. Theme visuel et ergonomie

### Question
Pourquoi avoir travaille aussi le theme et l'ergonomie de l'application ?

### Reponse
Une application analytique ne doit pas seulement fonctionner techniquement. Elle doit aussi etre lisible, coherente et agreable a utiliser. Nous avons donc harmonise le theme Streamlit et le CSS custom afin d'avoir une interface sombre, lisible et compatible avec les cartes, graphiques et tableaux de bord.

L'objectif etait d'ameliorer la comprehension des resultats et la qualite generale de la demonstration.

### Emplacements dans le code
- [.streamlit/config.toml](.streamlit/config.toml#L1)
- [assets/style.css](assets/style.css#L3)
- [assets/style.css](assets/style.css#L26)

---

## 25. Limites du projet

### Question
Quelles sont selon vous les principales limites du projet ?

### Reponse
Les principales limites sont les suivantes :

- VADER reste limite sur le sarcasme, l'ironie et certaines nuances complexes ;
- DistilBERT n'est pas fine-tune sur un corpus textile specifique ;
- l'analyse par aspects repose sur des mots-cles et regles metier ;
- le topic modeling devient moins stable quand le corpus final est petit ;
- le scraping reste fragile face aux protections anti-bot.

Ces limites sont normales dans un prototype academique, mais elles ouvrent de bonnes perspectives d'amelioration.

### Emplacements dans le code
- [models/sentiment_vader.py](models/sentiment_vader.py#L47)
- [models/sentiment_bert.py](models/sentiment_bert.py#L15)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L448)
- [models/topic_model.py](models/topic_model.py#L64)
- [scraping/shein_scraper.py](scraping/shein_scraper.py#L130)

---

## 26. Reponse de synthese si le professeur demande le point fort du projet

### Question
Quel est selon vous le principal point fort technique de votre projet ?

### Reponse
Le principal point fort du projet est son integration de bout en bout. Nous n'avons pas seulement implemente un modele de sentiment. Nous avons construit une application complete capable de charger des donnees heterogenes, de les nettoyer, de les analyser avec plusieurs briques NLP, de produire des KPI metier, de generer des recommandations et de proposer une interface exploitable.

Autrement dit, la valeur du projet vient autant de l'architecture globale et de la robustesse du pipeline que des modeles eux-memes.

### Emplacements dans le code
- [app.py](app.py#L238)
- [models/nlp_pipeline.py](models/nlp_pipeline.py#L39)
- [utils/kpi_calculator.py](utils/kpi_calculator.py#L14)
- [models/decision_engine.py](models/decision_engine.py#L17)

---

## 27. Reponse de synthese si le professeur demande ce que vous ajouteriez ensuite

### Question
Si vous aviez plus de temps, quelles seraient vos prochaines ameliorations ?

### Reponse
Les evolutions prioritaires seraient :

- fine-tuning d'un modele Transformer sur un corpus metier ;
- enrichissement automatique des aspects ;
- evaluation sur un dataset annote manuellement ;
- stockage persistant des analyses ;
- alertes automatiques en temps reel ;
- meilleure industrialisation du scraping.

Cette reponse montre que nous connaissons les limites actuelles et que nous avons une vision claire de la suite.

### Emplacements dans le code
- [models/sentiment_bert.py](models/sentiment_bert.py#L15)
- [models/aspect_analyzer.py](models/aspect_analyzer.py#L249)
- [utils/metrics.py](utils/metrics.py#L163)
- [scraping/scraper_pipeline.py](scraping/scraper_pipeline.py#L141)
