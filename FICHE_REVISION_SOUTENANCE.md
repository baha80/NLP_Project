# Fiche de Revision - Soutenance Technique

Cette fiche resume les questions techniques les plus probables, avec des reponses courtes de revision.

---

## 1. Pourquoi avoir choisi une architecture modulaire ?

Nous avons separe le projet en modules pour isoler les responsabilites : preprocessing, sentiment, NER, aspects, topics, KPI, decisions et scraping. Cela rend le code plus lisible, plus maintenable et plus facile a faire evoluer. Le systeme multipage permet aussi de separer les usages dans l'interface.

Code : [app.py](app.py#L97), [models/nlp_pipeline.py](models/nlp_pipeline.py#L19)

---

## 2. Quel est le pipeline global de l'application ?

Le pipeline commence par le chargement et la normalisation des donnees, puis il applique le nettoyage, la tokenisation, VADER, le NER, TF-IDF, SVD, l'analyse par aspects, le topic modeling, le calcul des KPI et enfin le moteur de decision. Les resultats sont ensuite stockes dans `st.session_state` pour etre affiches dans les pages Streamlit.

Code : [app.py](app.py#L238), [models/nlp_pipeline.py](models/nlp_pipeline.py#L39)

---

## 3. Comment avez-vous gere les CSV heterogenes ?

Nous avons ajoute un systeme d'alias pour reconnaitre plusieurs noms de colonnes possibles comme `review`, `comment`, `content`, `stars` ou `score`. Si certaines colonnes manquent, l'application ajoute des valeurs par defaut. Cela rend l'import beaucoup plus robuste pour des datasets reels ou Kaggle.

Code : [app.py](app.py#L34), [app.py](app.py#L186)

---

## 4. Que fait le preprocessing ?

Le preprocessing nettoie les textes avant analyse. Il supprime URL, HTML, emojis, ponctuation, espaces inutiles et stopwords, puis tokenise le texte. Il retire aussi les avis vides et les doublons, ce qui ameliore fortement la qualite des resultats.

Code : [preprocessing/cleaner.py](preprocessing/cleaner.py#L53), [preprocessing/cleaner.py](preprocessing/cleaner.py#L89)

---

## 5. Pourquoi le nombre d'avis baisse-t-il apres nettoyage ?

Parce qu'on retire les doublons, les lignes vides et les textes trop faibles. Sur notre dataset principal, on passe de 500 lignes brutes a 32 avis reellement exploitables. Cela montre que la preparation des donnees est essentielle pour eviter une analyse biaisee par le bruit.

Code : [preprocessing/cleaner.py](preprocessing/cleaner.py#L89)

---

## 6. Pourquoi VADER comme modele principal ?

VADER est rapide, interpretable et adapte aux avis courts. Il ne demande pas d'entrainement et fournit directement un score positif, negatif, neutre et compound. C'est un excellent choix pour une application interactive qui doit rester fluide.

Code : [models/sentiment_vader.py](models/sentiment_vader.py#L47), [models/sentiment_vader.py](models/sentiment_vader.py#L55)

---

## 7. Pourquoi DistilBERT est-il optionnel ?

DistilBERT est plus riche contextuellement, mais il est plus lourd et plus lent a charger. Nous l'avons garde comme modele de comparaison pour enrichir la demonstration sans rendre toute l'application dependante d'un modele Transformer. Cela garde un bon compromis entre performance et praticite.

Code : [models/sentiment_bert.py](models/sentiment_bert.py#L15), [models/sentiment_bert.py](models/sentiment_bert.py#L72)

---

## 8. Comment avez-vous gere la classe neutre avec DistilBERT ?

Le modele utilise est essentiellement binaire, donc nous avons ajoute une logique basee sur le score de confiance. Quand la confiance est trop faible, le systeme transforme la prediction en `Neutral`. Cela evite de forcer une polarite forte sur des avis ambigus.

Code : [models/sentiment_bert.py](models/sentiment_bert.py#L23), [models/sentiment_bert.py](models/sentiment_bert.py#L56)

---

## 9. Comment fonctionne le NER ?

Le systeme essaie d'abord de charger spaCy avec `en_core_web_sm`. Si le modele n'est pas disponible, il utilise une extraction regex de secours. Cette approche garantit que le pipeline continue a fonctionner meme si toutes les ressources NLP ne sont pas installees.

Code : [models/ner.py](models/ner.py#L14), [models/ner.py](models/ner.py#L111)

---

## 10. Pourquoi TF-IDF puis TruncatedSVD ?

TF-IDF donne une representation simple et robuste des textes. TruncatedSVD compresse ensuite cette representation en vecteurs denses plus compacts. Cette approche est legere, interpretable et bien adaptee a un projet academique local.

Code : [preprocessing/vectorizer.py](preprocessing/vectorizer.py#L50), [preprocessing/vectorizer.py](preprocessing/vectorizer.py#L121)

---

## 11. Pourquoi avoir ajoute une analyse par aspects ?

Le sentiment global ne dit pas pourquoi un client est satisfait ou insatisfait. L'analyse par aspects permet de lier l'opinion a une dimension metier precise comme la qualite, la taille, la livraison, le prix ou le design. Cela rend les resultats beaucoup plus utiles pour la decision.

Code : [models/aspect_analyzer.py](models/aspect_analyzer.py#L249), [models/aspect_analyzer.py](models/aspect_analyzer.py#L260)

---

## 12. Pourquoi des regles metier dans les aspects ?

Certaines expressions comme `runs small`, `too tight` ou `overpriced` ont un sens metier tres clair. Nous avons ajoute des regles d'override pour mieux corriger la polarite dans ces cas frequents. Cela ameliore la precision pratique et l'interpretation des resultats.

Code : [models/aspect_analyzer.py](models/aspect_analyzer.py#L448)

---

## 13. Pourquoi du topic modeling en plus des aspects ?

Les aspects sont predefinis, alors que le topic modeling sert a faire emerger automatiquement des themes recurrents. Les deux approches sont complementaires : les aspects repondent aux questions metier connues, tandis que LDA aide a decouvrir des sujets inattendus dans le corpus.

Code : [models/topic_model.py](models/topic_model.py#L64), [models/topic_model.py](models/topic_model.py#L111)

---

## 14. Comment avez-vous valide le modele ?

Nous avons compare les predictions VADER aux sentiments derives des notes clients. Les notes 4-5 sont mappees en positif, 3 en neutre, 1-2 en negatif. Ensuite, nous calculons accuracy, precision, recall, F1 et matrice de confusion.

Code : [utils/metrics.py](utils/metrics.py#L140), [utils/metrics.py](utils/metrics.py#L163)

---

## 15. Quels KPI avez-vous implemente ?

Nous avons calcule des KPI metier comme Overall Customer Satisfaction, Quality Score, Return Risk Index, Delivery Score, Net Promoter Sentiment et Review Volume. L'objectif est de transformer des avis textuels en indicateurs directement exploitables par un decideur.

Code : [utils/kpi_calculator.py](utils/kpi_calculator.py#L14), [pages/1_Dashboard.py](pages/1_Dashboard.py#L35)

---

## 16. Comment fonctionne le moteur de decision ?

Le moteur applique des seuils sur les aspects et les tendances. Si les plaintes de sizing ou de qualite depassent un seuil, il genere une recommandation priorisee. Il peut aussi detecter des hausses soudaines dans les topics pour alerter sur un probleme emergent.

Code : [models/decision_engine.py](models/decision_engine.py#L17), [models/decision_engine.py](models/decision_engine.py#L153)

---

## 17. Pourquoi utiliser st.session_state ?

`st.session_state` permet de conserver les donnees et les resultats entre les pages. L'analyse est lancee une seule fois, puis toutes les pages reutilisent les memes objets. Cela evite les recalculs inutiles et rend l'application plus fluide.

Code : [app.py](app.py#L110), [app.py](app.py#L316)

---

## 18. Que montre la page d'analyse temps reel ?

Elle permet d'analyser un avis unique, un petit batch de textes ou de comparer VADER et DistilBERT. C'est une page tres utile en soutenance parce qu'elle montre le comportement du systeme en direct, sans passer par tout un dataset.

Code : [pages/2_Analyze.py](pages/2_Analyze.py#L57), [pages/2_Analyze.py](pages/2_Analyze.py#L60)

---

## 19. A quoi sert la page NER ?

La page NER permet d'explorer les entites extraites, leurs frequences, leurs labels et les avis dans lesquels elles apparaissent. Elle ajoute une lecture qualitative complementaire au simple sentiment global et aide a repérer des marques, produits ou lieux cites par les clients.

Code : [pages/6_NER.py](pages/6_NER.py#L41), [pages/6_NER.py](pages/6_NER.py#L57)

---

## 20. Pourquoi avoir separe le scraping de l'application principale ?

Le scraping est plus fragile que le pipeline NLP, car il depend de sites externes, de leur HTML et de leurs protections anti-bot. En le gardant dans des modules et scripts dedies, nous evitons que l'application analytique soit bloquee quand une plateforme change ou se protege davantage.

Code : [scraping/scraper_pipeline.py](scraping/scraper_pipeline.py#L28), [scraping/scraper_pipeline.py](scraping/scraper_pipeline.py#L141)

---

## 21. Comment fonctionne le scraper Amazon ?

Le scraper Amazon extrait l'ASIN d'une URL ou reference produit, puis charge les pages d'avis avec Selenium. Il peut aussi ouvrir un navigateur en mode manuel pour laisser l'utilisateur resoudre un CAPTCHA avant la collecte. Cette strategie est necessaire face aux protections Amazon.

Code : [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L126), [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L293), [scraping/amazon_scraper.py](scraping/amazon_scraper.py#L301)

---

## 22. Pourquoi un mode manuel pour SHEIN ?

SHEIN declenche souvent des challenges de securite qui bloquent les requetes simples. Nous avons donc implemente un scraper navigateur avec detection de challenge, pause manuelle et extraction du DOM charge. C'est une solution pragmatique face aux contraintes du web reel.

Code : [scraping/shein_scraper.py](scraping/shein_scraper.py#L130), [scraping/shein_scraper.py](scraping/shein_scraper.py#L155), [scraping/shein_scraper.py](scraping/shein_scraper.py#L436)

---

## 23. Quel est le principal point fort du projet ?

Le point fort principal est l'integration de bout en bout. Nous n'avons pas seulement un modele de sentiment : nous avons un systeme complet avec import, nettoyage, NLP, KPI, recommandations, dashboard et scraping. La valeur vient donc autant de l'architecture globale que des modeles eux-memes.

Code : [app.py](app.py#L238), [models/decision_engine.py](models/decision_engine.py#L17)

---

## 24. Quelles sont les principales limites du projet ?

Les limites principales sont les suivantes : VADER reste limite sur les nuances complexes, DistilBERT n'est pas fine-tune sur le domaine textile, les aspects reposent sur des mots-cles, LDA depend de la taille du corpus, et le scraping reste fragile face aux anti-bots. Ces limites sont normales pour un prototype academique.

Code : [models/sentiment_vader.py](models/sentiment_vader.py#L47), [models/aspect_analyzer.py](models/aspect_analyzer.py#L448), [scraping/shein_scraper.py](scraping/shein_scraper.py#L130)

---

## 25. Que pourriez-vous ajouter dans une version future ?

Les ameliorations naturelles seraient un fine-tuning d'un Transformer metier, une meilleure evaluation sur un corpus annote, une base de donnees pour historiser les analyses, des alertes automatiques et une industrialisation plus robuste du scraping. Cela montre que nous avons une vision claire des perspectives du projet.

Code : [models/sentiment_bert.py](models/sentiment_bert.py#L15), [utils/metrics.py](utils/metrics.py#L163), [scraping/scraper_pipeline.py](scraping/scraper_pipeline.py#L141)