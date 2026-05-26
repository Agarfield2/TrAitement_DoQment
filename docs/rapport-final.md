# TrAitement-DoQment

## Un système RAG local pour la gestion documentaire conforme à la Loi 25 du Québec

**Rapport final — 8INF934 Atelier pratique en intelligence artificielle I**

*BEHAREL Armand · JEANNE Arthur · SOUKI Mohamed*

*Université du Québec à Chicoutimi — Session d'hiver 2026*

---

## Résumé

Ce projet propose un système de gestion documentaire local capable de répondre à des questions en langage naturel sur un corpus de documents administratifs, sans qu'aucune donnée ne quitte la machine de l'utilisateur. Le système combine deux pipelines complémentaires de Retrieval-Augmented Generation : un pipeline textuel reposant sur l'OCR et les embeddings MPNet indexés dans FAISS, et un pipeline multimodal reposant sur les embeddings visuels ColQwen2 indexés dans Qdrant. La génération est assurée par Mistral 7B et Qwen2.5-VL 7B via Ollama, garantissant l'exécution intégralement hors-ligne. L'ensemble est validé par 63 tests automatisés s'exécutant en moins de trois secondes, et nous avons indexé 16 723 passages issus de 2 089 documents du corpus SROIE 2019. Le projet illustre comment trois bases de code livrées par les coéquipiers ont pu coexister sans modification grâce à une couche d'enrobage soigneusement conçue, et documente les défis pratiques rencontrés : bugs canoniques contournés, robustesse de la détection d'environnement, gestion des particularités d'un corpus public hétérogène.

---

## 1. Introduction

### 1.1 Contexte législatif et technique

Depuis l'entrée en vigueur progressive de la Loi 25 modernisant la protection des renseignements personnels au Québec, les organismes publics et les entreprises traitant de la donnée personnelle sont tenus de documenter les transferts hors-province et de désigner un responsable de la protection des renseignements personnels. Or, la majorité des outils de traitement documentaire assistés par intelligence artificielle reposent aujourd'hui sur des services infonuagiques opérés depuis les États-Unis ou l'Europe, ce qui complique singulièrement la conformité à cette loi : chaque requête envoyée à un grand modèle de langage commercial déplace de fait le contenu du document à l'étranger, déclenchant les obligations de l'article 17 de la Loi sur l'accès aux documents des organismes publics et de l'article 70.1 de la Loi sur la protection des renseignements personnels dans le secteur privé.

Notre projet répond à ce constat en construisant un système qui n'envoie strictement rien sur le réseau une fois l'installation faite. Cette contrainte d'exécution locale n'est pas qu'une commodité réglementaire : elle structure l'intégralité des choix techniques que nous avons effectués, depuis le choix des modèles jusqu'à la stratégie d'indexation, en passant par les compromis sur la qualité des réponses.

### 1.2 Problématique

La question centrale que nous avons cherché à résoudre s'énonce simplement : *comment construire, avec les outils disponibles en 2025-2026, un système de questions-réponses sur documents qui soit suffisamment performant pour être utile, mais entièrement exécutable sur du matériel grand public sans connexion internet ?* Cette question recoupe trois sous-problèmes que nous traitons séparément : l'extraction du contenu (OCR sur les images, parsing des PDF), la recherche pertinente dans une base potentiellement grande, et la génération de réponses citées et fidèles aux sources.

### 1.3 Plan du rapport

Après un survol des choix technologiques et de leurs justifications (section 2), nous décrivons l'architecture globale du système (section 3) puis chacun des deux pipelines en détail (sections 4 et 5). La section 6 explique notre organisation du code, notamment la cohabitation entre les contributions des trois coéquipiers, conservées byte-identiques. La section 7 raconte les défis techniques effectivement rencontrés pendant l'implémentation et comment nous les avons résolus. Les sections 8 et 9 présentent la méthodologie de validation et les résultats expérimentaux. La section 10 documente les limites assumées du système avant la conclusion en section 11.

---

## 2. État de l'art et choix technologiques

### 2.1 Le paradigme RAG

L'approche Retrieval-Augmented Generation, formalisée par Lewis et collaborateurs en 2020, consiste à conditionner un modèle de langage sur des passages préalablement récupérés d'une base documentaire, plutôt que de s'en remettre à la seule connaissance paramétrique du modèle. Ce paradigme présente trois vertus particulièrement intéressantes dans notre contexte. Premièrement, il permet d'utiliser un modèle de génération relativement petit sans sacrifier la précision factuelle, puisque l'information de référence est fournie en contexte. Deuxièmement, il rend les réponses traçables : chaque affirmation peut être attribuée à un passage source, ce qui est essentiel pour un usage administratif où la vérifiabilité prime. Troisièmement, il découple la mise à jour des données (réindexation) de la mise à jour du modèle, ce qui simplifie considérablement la maintenance.

### 2.2 RAG textuel et RAG multimodal

La littérature distingue habituellement deux familles d'approches RAG selon la nature des représentations utilisées. Le RAG textuel passe par une étape d'extraction de texte (OCR pour les images, parsing pour les PDF) suivie d'un encodage par un modèle d'embeddings de phrases. Cette approche est mature, rapide à l'inférence et bien outillée, mais elle perd l'information de mise en page : tableaux, formulaires structurés, graphiques deviennent des soupes de mots dont l'ordre dépend de la qualité de l'OCR. Le RAG multimodal, au contraire, encode directement chaque page comme une image et confie la lecture du document au modèle de génération multimodal au moment de la requête. Cette approche, popularisée par les travaux sur ColPali en 2024, préserve toute l'information visuelle mais demande beaucoup plus de calcul et de mémoire.

Plutôt que de choisir entre les deux, nous avons implémenté les deux en parallèle. L'utilisateur choisit lequel utiliser selon la nature de ses documents : factures et reçus structurés se prêtent bien au pipeline multimodal, courriers administratifs et rapports en prose se prêtent bien au pipeline textuel.

### 2.3 Choix des modèles d'embeddings

Pour le pipeline textuel, nous utilisons `sentence-transformers/all-mpnet-base-v2`, le modèle phare des embeddings de phrases multilingues. Il produit des vecteurs de 768 dimensions normalisés en norme 2, ce qui transforme la similarité cosinus en simple produit scalaire et accélère la recherche. La fenêtre de contexte limitée du modèle (512 tokens) n'est pas un problème dans notre cas puisque les passages sont déjà découpés finement par l'étape de chunking.

Pour le pipeline multimodal, nous utilisons `vidore/colqwen2-v1.0`. Ce modèle, dérivé de Qwen2-VL par fine-tuning sur ColPali, produit non pas un vecteur unique par page mais une matrice de vecteurs, un par patch visuel. La recherche se fait alors par une opération de *max similarity* : pour chaque token de la requête, on prend le patch le plus similaire dans chaque page candidate, et on somme. Cette représentation préserve l'information locale et est nettement plus précise sur les documents structurés, au prix d'un stockage et d'un calcul plus lourds (typiquement 1024 patchs de 128 dimensions par page).

### 2.4 Choix des index

FAISS est devenu le standard de fait pour l'indexation vectorielle dense, et son implémentation HNSW (Hierarchical Navigable Small World) offre un excellent compromis entre vitesse de construction, vitesse de recherche et qualité du rappel. Nous l'utilisons pour le pipeline textuel.

Pour le pipeline multimodal, le besoin est différent : chaque page produit non pas un vecteur mais une matrice, et l'opération de similarité est la max-sim plutôt que le produit scalaire. Nous utilisons Qdrant en mode *local path*, qui supporte nativement les vecteurs multiples par document via son interface *multi-vector*. Le mode local-path nous évite d'avoir à lancer un serveur Qdrant séparé, ce qui simplifie le déploiement.

### 2.5 Choix du backend de génération

Nous avons fait le choix d'un backend unique pour les deux pipelines : Ollama. Cette décision a beaucoup d'avantages pratiques. Ollama gère le téléchargement et la quantification des modèles, expose une API HTTP unifiée pour les générations texte et vision, et s'installe en une commande sur les trois systèmes d'exploitation cibles. Le modèle textuel choisi est `mistral:7b-instruct` pour son excellent rapport qualité-taille et sa capacité francophone honorable. Le modèle visuel choisi est `qwen2.5vl:7b`, qui en 2025-2026 reste la référence open-weights pour les tâches de question-réponse multimodale, devant LLaVA et MiniCPM-V à mémoire égale.

Nous avons explicitement renoncé à supporter d'autres backends (llama.cpp directement, transformers en local, OpenAI compatible). Cette simplification réduit la surface d'API à maintenir et oblige à un seul chemin de tests.

---

## 3. Architecture du système

### 3.1 Vue d'ensemble

Le système se structure en quatre couches concentriques. Au cœur, cinq fichiers Python conservés byte-identiques aux livraisons des coéquipiers (`ingestion.py`, `pipeline1.py`, `pdf_ocr.py`, `Comparaison_OCR.py`, `pipeline_rag2.py`) fournissent les briques de base : OCR, chunking, classes d'embeddings, indexation FAISS. Autour de ce noyau, le package `doqment/` enrobe ces briques en huit modules cohérents qui exposent une API publique propre, masquent les bugs résiduels du noyau et ajoutent les fonctionnalités manquantes. Au-dessus, le dossier `scripts/` fournit deux interfaces en ligne de commande (`phase1.py` et `phase2.py`) construites avec Typer. Enfin, l'application Streamlit `app.py` répartit les requêtes vers les vues spécialisées du sous-package `doqment.ui/`.

```
┌───────────────────────────────────────────────────────────┐
│  app.py (Streamlit dispatcher, 50 lignes)                  │
│  ↓                                                          │
│  doqment/ui/{doc_view, db_view, ingest_view}.py            │
│  scripts/phase1.py · scripts/phase2.py  (Typer CLIs)       │
│  ↓                                                          │
│  doqment/                                                   │
│    settings.py · llm.py · ocr.py                            │
│    phase1.py · phase2.py · phase2_store.py                  │
│  ↓                                                          │
│  ingestion.py · pipeline1.py · pdf_ocr.py                   │
│  Comparaison_OCR.py · pipeline_rag2.py    ← byte-identique  │
└───────────────────────────────────────────────────────────┘
```

Cette stratification est volontaire : aucun module supérieur n'écrit dans un module inférieur, et les fichiers canoniques au sol ne sont jamais touchés. Le risque résiduel est entièrement absorbé par les wrappers de la couche `doqment/`.

### 3.2 Configuration centralisée

Tous les paramètres ajustables du système sont rassemblés dans une seule dataclass `Settings` exposée par `doqment.settings`. Chaque champ a une valeur par défaut et peut être surchargé par une variable d'environnement préfixée `DOQMENT_`. Par exemple, le modèle de vision se règle via `DOQMENT_OLLAMA_VISION_MODEL`, le chemin du binaire Tesseract via `DOQMENT_TESSERACT_PATH`, le périphérique de ColQwen2 via `DOQMENT_COLQWEN_DEVICE`. Nous avons délibérément évité d'utiliser `pydantic-settings` ou un fichier YAML : la dataclass standard suffit, sans dépendance supplémentaire, et le débogage par `print(settings)` reste lisible.

### 3.3 Trois modes d'interaction, deux pipelines

L'utilisateur dispose de six points d'entrée que nous présentons comme une matrice à deux dimensions. Selon le pipeline (textuel ou multimodal) et le mode (`ingest`, `doc`, `db`), il invoque l'une des combinaisons suivantes :

| Mode | Pipeline 1 (textuel) | Pipeline 2 (multimodal) |
|---|---|---|
| **`ingest`** | Construit l'index FAISS depuis un dossier | Construit l'index Qdrant depuis un dossier |
| **`doc`** | Réponse en mémoire sur **un** document | Réponse en mémoire sur **un** document |
| **`db`** | Réponse contre la base FAISS complète | Réponse contre la base Qdrant complète |

Le mode `doc` n'utilise jamais l'index persistant ; il rasterise le document, encode les passages ou les pages à la volée, applique la recherche en mémoire et appelle directement le LLM. Cette distinction est importante : elle permet à un utilisateur de poser une question sur un nouveau document sans avoir à le verser au préalable dans la base, et inversement de poser une question sur la base sans avoir besoin du document spécifique.

---

## 4. Pipeline 1 — RAG textuel

### 4.1 Chaîne de traitement

Le pipeline textuel suit la chaîne canonique d'un RAG dense : extraction du texte, découpage en passages, encodage, indexation, recherche, génération. Les briques individuelles proviennent du fichier `ingestion.py` du coéquipier 1, qui implémente correctement le chunking par regroupement géométrique de boîtes de texte et la classe `FAISSIndex` avec sauvegarde sur disque. Notre rôle dans `doqment/phase1.py` consiste à orchestrer ces briques sans en réimplémenter aucune.

L'extraction du texte se fait selon deux voies. Si une annotation au format ICDAR (lignes `x1,y1,...,x4,y4,transcription` du jeu SROIE) est disponible à côté de l'image, nous la passons directement au parser canonique `OCREngine.from_sroie_annotation`. Sinon, nous passons l'image dans notre wrapper `doqment.ocr.ocr_image` qui appelle Tesseract avec un prétraitement adaptatif. Cette double voie nous permet à la fois d'ingérer rapidement le corpus SROIE annoté et de traiter des documents arbitraires sans annotation.

### 4.2 Embeddings et indexation

Une fois les passages extraits, ils sont encodés par lots via `EmbeddingModel.encode`, qui charge `all-mpnet-base-v2` à la première utilisation et le maintient en mémoire ensuite. Les vecteurs sont accumulés puis insérés dans un index FAISS HNSW dont les paramètres (`M=16`, `efSearch=200`) sont ceux de la livraison canonique. L'index et ses métadonnées sont sauvegardés dans `data/processed/`.

### 4.3 Recherche et génération

À la requête, la même classe `EmbeddingModel` encode la question, et la méthode `FAISSIndex.search` du canonique renvoie une liste de dictionnaires contenant le texte du passage, le fichier source, la page, la confiance moyenne d'OCR et le score de similarité. Notre wrapper effectue ensuite un dédoublonnage par triplet *(nom de fichier, page, texte)* avant de construire le prompt — sans cette étape, les réponses sont polluées par les doublons que le corpus SROIE introduit en répartissant les mêmes images dans `task1train` et `task1&2_test`.

Le prompt construit suit le format `[INST] ... [/INST]` de Mistral, avec une instruction explicite de citer chaque passage par son numéro entre crochets et un message de refus précis (« I do not have this information in the provided documents. ») à utiliser si la réponse n'est pas trouvable. Ce message est ensuite contrôlé par les tests pour s'assurer qu'il n'est pas dilué par le modèle.

---

## 5. Pipeline 2 — RAG multimodal

### 5.1 Encodage visuel par ColQwen2

Le pipeline multimodal commence par une étape de rastérisation : chaque PDF est converti en images de pages individuelles via `pdf2image` (qui invoque `poppler-utils` en interne). Les images obtenues, à 200 DPI par défaut, sont ensuite passées dans le modèle ColQwen2 par la classe `ColQwen2Encoder` que nous avons écrite dans `doqment.phase2_store`. Ce modèle, contrairement aux encodeurs de phrases classiques, produit pour chaque image une matrice de typiquement 1024 vecteurs de 128 dimensions — un par patch visuel de l'image.

### 5.2 Indexation multi-vecteurs dans Qdrant

Qdrant supporte nativement les *multi-vector embeddings* via sa fonctionnalité `VectorParams` paramétrée en mode `MAX_SIM`. Concrètement, chaque page de document est insérée comme un point unique dont le payload contient son chemin sur disque et son numéro de page, et dont l'embedding est la matrice de vecteurs produite par ColQwen2. La recherche est entièrement déléguée à Qdrant qui calcule la max-sim entre la matrice de requête et chaque matrice candidate.

Le store complet est complété par une classe `MetadataStore` que nous avons ajoutée, qui maintient un fichier JSON listant les documents déjà indexés, leurs hachages et leurs nombres de pages. Cette information est consultée à chaque ingestion pour ignorer les documents déjà présents — l'opération est ainsi *idempotente*, ce qui est testé.

### 5.3 Génération multimodale avec Qwen2.5-VL

Au moment de la requête, ColQwen2 encode la question textuelle, Qdrant retourne les meilleures pages, et les images de ces pages sont envoyées avec la question à Qwen2.5-VL via l'API Ollama. Le prompt système demande au modèle de répondre au format JSON strict : `{"answer": "...", "cited_pages": [0, 2]}`. Cette structure permet au wrapper de vérifier que les indices cités sont bien dans la plage des pages soumises, et de propager au final un message de refus francophone (« Information non trouvée dans les documents fournis. ») si le JSON est mal formé ou si aucune page n'est citée.

### 5.4 Compromis de mémoire et de temps

Le pipeline 2 est nettement plus coûteux que le pipeline 1. Qwen2.5-VL 7B nécessite environ 12,5 Go de RAM en exécution CPU-only — 5 Go avec un GPU CUDA — auxquels s'ajoutent environ 6 Go pour ColQwen2 chargé en mémoire. Sur les machines de test des membres de l'équipe (Fedora 44, 16 Go de RAM, sans GPU dédié), l'usage est marginal et impose de fermer les autres applications avant le lancement. Nous documentons cette contrainte dans le `README.md` et l'assumons : dégrader le modèle vision à une variante plus petite aurait significativement détérioré la qualité des réponses sur les documents structurés, ce qui était précisément la raison d'exister du pipeline 2.

---

## 6. Organisation du code et fichiers canoniques

### 6.1 Cinq fichiers byte-identiques aux livraisons

Une particularité méthodologique de ce projet mérite d'être détaillée. Au démarrage du travail d'intégration, nous avions trois bases de code livrées par les coéquipiers : un module d'ingestion et un pipeline RAG textuel du coéquipier 1 (`ingestion.py`, `pipeline1.py`), un système RAG avec interface REPL du coéquipier 2 (`pipeline_rag2.py`), et deux utilitaires d'OCR (`pdf_ocr.py`, `Comparaison_OCR.py`). Plutôt que de fusionner ces fichiers en réécrivant l'historique, nous avons fait le choix de les conserver tels quels et de les invoquer depuis notre couche d'enrobage.

Cette décision a plusieurs justifications. D'abord, elle préserve la traçabilité du travail individuel : chaque coéquipier peut pointer vers son fichier d'origine pour défendre sa contribution. Ensuite, elle nous oblige à ne pas réinventer ce qui marche déjà : le chunking géométrique par regroupement de boîtes, la sauvegarde-rechargement FAISS, l'extraction des annotations ICDAR sont des fonctions canoniques que nous appelons telles quelles. Enfin, elle nous force à expliciter par des wrappers les éventuels défauts qui subsistent, plutôt que de les corriger localement et d'introduire des divergences silencieuses.

Concrètement, à chaque session de travail nous vérifions par un `diff` automatisé que ces cinq fichiers sont strictement identiques aux versions de référence stockées dans des dossiers à part. Cette vérification fait partie de notre pipeline de packaging et nous a permis de détecter une fois une modification accidentelle.

### 6.2 Le package `doqment/`

Au-dessus du noyau canonique, le package `doqment/` contient sept modules totalisant environ 1 970 lignes de Python. Le module `settings.py` (72 lignes) expose la dataclass de configuration. Le module `llm.py` (184 lignes) fournit deux fonctions `generate_text` et `generate_vision` au-dessus de l'API Ollama, avec gestion d'erreurs spécifique. Le module `ocr.py` (233 lignes) enrobe Tesseract via `pytesseract` avec détection robuste du binaire, prétraitement adaptatif et regroupement des mots en lignes au format `ingestion.TextLine`. Les modules `phase1.py` (530 lignes) et `phase2.py` (346 lignes) implémentent les trois fonctions publiques de chaque pipeline (`ingest_directory`, `ask_document`, `ask_database`). Le module `phase2_store.py` (459 lignes) regroupe les classes spécifiques au pipeline 2 : `ColQwen2Encoder`, `QdrantStore`, `MetadataStore` et les fonctions utilitaires de rastérisation et de calcul de max-sim. Le sous-package `doqment/ui/` contient enfin les trois vues Streamlit.

### 6.3 Tests et garde-fous

Le dossier `tests/` contient 63 tests répartis en six fichiers, totalisant environ 1 330 lignes. La majorité de ces tests sont des tests unitaires rapides qui exercent une fonction isolée avec des mocks pour les dépendances externes (LLM, embeddings, FAISS, Qdrant). Une poignée de tests utilisent les vraies classes canoniques (`FAISSIndex`, `EmbeddingModel`) afin de valider que notre wrapper et le noyau communiquent bien sur les formats attendus — ces tests sont essentiels car ils auraient attrapé certains des bugs que nous avons rencontrés en production et que nous décrivons en section 7.

L'ensemble de la suite s'exécute en moins de trois secondes sur une machine standard, ce qui rend la boucle de développement très courte et encourage à faire tourner les tests à chaque modification.

---

## 7. Défis techniques et décisions de conception

Cette section décrit cinq problèmes techniques effectivement rencontrés pendant l'implémentation et nos solutions. Plutôt que de présenter une vue lissée du projet, nous documentons ici les détours et les fausses pistes, parce qu'ils éclairent les choix qui peuvent paraître arbitraires dans la version finale.

### 7.1 Le bug Typer : un flag booléen interprété comme une chaîne

La fonction `ingest_directory` de notre wrapper accepte un paramètre `use_tesseract` qui contrôle si on doit OCR-iser les images sans annotation. Notre première version exposait ce paramètre comme une option de la ligne de commande Typer ainsi : `use_paddle=typer.Option(False, "--paddle", help="...")`. Sans annotation de type explicite, Typer infère le type à partir du défaut, mais avec une subtilité qui nous a piégés : la valeur transmise au callback est alors la chaîne `"False"`, pas le booléen `False`. Or `bool("False")` vaut `True` en Python parce que toute chaîne non-vide est *truthy*. Conséquence : sans aucun flag sur la ligne de commande, le paramètre `use_paddle_ocr` du constructeur canonique recevait `True`, ce qui engageait silencieusement la branche PaddleOCR — laquelle est cassée dans le fichier canonique.

La correction a tenu en deux mesures. D'abord, annoter explicitement le type de toutes les options Typer (`use_tesseract: bool`, `top_k: int`, `task2: Optional[str]`), ce qui force Typer à utiliser sa logique de flag booléen avec l'inverse automatique (`--tesseract / --no-tesseract`). Ensuite, ajouter quatre tests dédiés dans `tests/test_cli.py` qui vérifient que les valeurs reçues par le wrapper sont bien des `bool`, `int` et `str` Python natifs, pas des chaînes. Si quelqu'un retire les annotations dans le futur, ces tests échouent immédiatement.

### 7.2 Le format de métadonnées FAISS : un dict supposé, une liste réelle

Notre première implémentation de `ask_database` reconstruisait le chargement de l'index FAISS à la main : ouvrir `metadata.pkl`, en extraire les passages, instancier l'index. Cette approche supposait, sans vérification, que le fichier picklé était un dictionnaire de la forme `{"passages": [...], "version": 1}`. Or le canonique sauvegarde directement la liste des dictionnaires de passages, sans enveloppe. À la première requête contre la base, le code échouait avec `TypeError: list indices must be integers or slices, not str`.

La correction a consisté à abandonner notre reconstruction maison et à utiliser à la place les méthodes `FAISSIndex.load` et `FAISSIndex.search` du canonique, qui font déjà tout le travail correctement et retournent des dictionnaires propres. En accompagnement, nous avons refait le test d'intégration `test_ask_database_returns_answer_when_index_present` pour qu'il construise un mini-index avec la vraie classe `FAISSIndex` (pas notre format imaginé), puis le charge et le requête. Ce test aurait attrapé le bug immédiatement s'il avait été écrit ainsi dès le départ.

### 7.3 La détection de Tesseract dans les terminaux à PATH minimal

L'un des membres de l'équipe travaille principalement depuis le terminal intégré d'Obsidian, qui exécute les commandes dans un environnement où la variable `PATH` ne contient pas `/usr/bin`. Conséquence : `shutil.which("tesseract")` renvoie `None` alors même que `dnf list installed` confirme la présence de `tesseract-5.5.2-1.fc44.x86_64` à `/usr/bin/tesseract`. Notre première détection levait alors une `RuntimeError` avec un message trompeur (« binaire non installé ») malgré l'installation correcte.

La solution déployée combine quatre niveaux de fallback. D'abord, une variable d'environnement `DOQMENT_TESSERACT_PATH` permet un override explicite. Ensuite, `shutil.which` est essayé. Si rien ne sort, une liste de chemins canoniques (`/usr/bin/tesseract`, `/usr/local/bin/tesseract`, `/opt/homebrew/bin/tesseract`, `/opt/local/bin/tesseract`) est testée par `os.path.isfile` et `os.access`. Enfin, si même cela échoue, nous laissons `pytesseract` essayer sa propre logique par défaut au moment de l'appel à `image_to_data` — c'est seulement si `pytesseract` lui-même lève `TesseractNotFoundError` que nous présentons un message d'erreur amical.

Cette stratification a un mérite épistémique : si un cas inattendu apparaît, l'erreur finale provient toujours de `pytesseract` lui-même, avec sa trace complète, plutôt que d'une erreur générique que nous aurions imposée trop tôt.

### 7.4 La classification des fichiers `.txt` du corpus SROIE

Le corpus SROIE 2019 contient cinq sous-dossiers dont les contenus se chevauchent : `0325updated.task1train(626p)` avec ses images et ses annotations ICDAR, `0325updated.task2train(626p)` avec ses entités JSON, mais aussi `task1&2_test`, `task3-test` et `text.task1&2-test` qui mélangent images et fichiers `.txt` de natures différentes. Notre première heuristique consistait à considérer comme annotation tout `.txt` situé dans le même dossier qu'une image de même radical. Cette règle marche pour `task1train`, mais elle échoue dans les dossiers de test où un `.txt` côte-à-côte d'un `.jpg` peut être en réalité du JSON, pas du format ICDAR. À l'ingestion, le parser ICDAR du canonique tentait alors de convertir `"address"` en flottant et plantait.

La correction a remplacé l'heuristique de localisation par une heuristique de contenu : nous classifions chaque `.txt` du corpus en regardant son premier caractère non-blanc. Si c'est `{` ou `[`, c'est du JSON (entité) ; sinon, c'est une annotation ICDAR. Cette classification est faite une seule fois en début d'ingestion, en parcourant l'arborescence récursivement, et produit deux dictionnaires indexés par radical. Le code de la boucle principale n'a plus qu'à faire deux lookups. Quatre tests dans `tests/test_phase1.py` couvrent les variations possibles, dont précisément le cas du JSON juxtaposé à une image qui nous avait fait crasher.

### 7.5 Le dédoublonnage des résultats de recherche

Une fois l'ingestion réparée et l'index construit avec 16 723 passages issus de 2 089 documents, nos premières requêtes ont produit des réponses étranges où le modèle citait `[1] [2] [3] [4]` comme s'il s'agissait de quatre sources distinctes, alors que les passages affichés étaient strictement identiques deux à deux. La cause se trouvait dans la structure du corpus : la même image `X51006913023.jpg` est présente à la fois dans `task1train` et dans `task1&2_test`, avec des chemins absolus différents mais un contenu OCR identique. FAISS, lui, retournait les deux occurrences avec des scores identiques.

La solution est un petit helper `_dedup_hits` qui collapse les doublons par triplet *(nom de fichier, page, texte)* avant la construction du prompt. Ce triplet est plus prudent qu'un dédoublonnage par texte seul (qui aurait pu coller deux passages identiques mais légitimement présents dans deux documents différents) et que par chemin seul (qui aurait gardé les doublons SROIE). Trois tests valident le comportement attendu : collapse des doublons train/test, conservation des pages différentes, conservation des textes différents.

---

## 8. Méthodologie de validation

### 8.1 Architecture de tests

La suite de tests compte 63 cas répartis comme suit : 4 tests CLI (vérification du typage Typer), 10 tests LLM (parsing des réponses, gestion des erreurs Ollama), 13 tests OCR (regroupement mots-lignes, détection du binaire, gestion des langues manquantes), 18 tests Pipeline 1 (extraction, embeddings, recherche, dédoublonnage, classification ICDAR/JSON), 14 tests Pipeline 2 (idempotence du store, max-sim, rastérisation), et 3 tests de configuration. La suite s'exécute en environ 2,5 secondes sur la machine de référence.

Trois principes guident l'écriture des tests. Premièrement, *aucune* dépendance réseau : les appels Ollama sont mockés via un client de remplacement, les embeddings HuggingFace via un faux encodeur 8-dimensionnel, et la classe Qdrant via un store en mémoire. Deuxièmement, lorsque la possibilité existe, on teste contre la *vraie* classe canonique plutôt que contre un faux : c'est notamment le cas du test d'intégration `FAISSIndex` qui sauvegarde puis recharge un index réel. Troisièmement, chaque bug rencontré en production donne lieu à au moins un nouveau test qui aurait permis de le détecter ; cette discipline a transformé chaque incident en garde-fou permanent.

### 8.2 Validation par exécution réelle

Au-delà des tests automatisés, nous validons régulièrement le système par des exécutions complètes sur le corpus SROIE. L'ingestion typique du corpus complet (2 154 images détectées dans `data/SROIE2019/`) prend environ trois minutes sur CPU, dont la majorité passée dans l'encodage MPNet par lots de 64 passages. La requête typique se complète en deux à cinq secondes après le chargement initial des modèles, qui prend environ dix secondes. Ces temps de réponse sont compatibles avec un usage interactif depuis l'interface Streamlit.

---

## 9. Résultats expérimentaux

### 9.1 Statistiques d'ingestion sur SROIE 2019

L'ingestion complète de `data/SROIE2019/` produit les chiffres suivants : 2 154 images au format `.jpg` détectées récursivement dans les cinq sous-dossiers du corpus, 1 196 fichiers `.txt` classifiés comme annotations ICDAR (premier caractère non-blanc numérique), 876 fichiers `.txt` classifiés comme entités JSON (premier caractère `{`). Sur les 2 154 images, 2 089 reçoivent une annotation ICDAR exploitable et sont ingérées ; 65 sont silencieusement ignorées car ni annotation ni `--tesseract` n'étaient disponibles. Le chunking canonique produit 16 723 passages à partir de ces 2 089 documents, soit en moyenne 8 passages par document.

### 9.2 Exemples de réponses du Pipeline 1

Sur une question typique « *What is the total amount?* » contre la base complète, le système retourne en deux secondes la réponse suivante (après dédoublonnage des sources) :

```
Question : What is the total amount?

Answer : The total amount in the document "X51006913023.jpg" (page 1) is $8.20.
The document "X51007846304.jpg" (page 1) reports RM 7.52 [2].

Sources (top 5) :
  [1] X51006913023.jpg p.1 score=0.577  TOTAL AMOUNT: $8.20
  [2] X51007846304.jpg p.1 score=0.607  TOTAL AMOUNT: RM7.52
  [3] X51005757286.jpg p.1 score=0.619  SUB TOTAL: 8.00 NET TOTAL : 8.00
```

Les scores de similarité sont calibrés : un score supérieur à 0,5 correspond à une correspondance forte sur le contenu du passage. Le modèle Mistral cite correctement les passages, et lorsqu'il manque d'information sur un document, il se replie sur le message de refus contrôlé.

### 9.3 Comportement du mode `doc`

Le mode `doc` permet de poser une question sur un document qui n'est pas dans la base. Sur une facture SROIE choisie au hasard (`X00016469612.jpg`) avec son annotation jumelle, la question « *What is the company name?* » produit en environ 1,5 seconde la réponse correcte : `TAN WOON YANN BOOK TA .K (TAMAN DAYA) SDN BHD`, avec le passage source cité explicitement. L'opération ne touche pas l'index persistant et n'écrit rien sur disque.

---

## 10. Limites assumées et perspectives

### 10.1 Limites techniques

Plusieurs limites du système sont assumées et documentées. L'index FAISS HNSW ne supporte pas la suppression individuelle de vecteurs ; pour retirer un document, il faut reconstruire l'index complet. Cette contrainte est conforme à la nature des structures HNSW, dont la performance dépend d'une construction monotone. Sur un usage administratif où les retraits sont rares, ce compromis est acceptable.

Le pipeline 2 consomme beaucoup de ressources : environ 18 Go de mémoire vive simultanée en exécution CPU-only, ce qui exige une machine de 24 Go de RAM minimale et la fermeture des autres applications avant le lancement. Sur du matériel équipé d'un GPU CUDA, ces chiffres se ramènent à environ 8 Go de VRAM, ce qui rentre confortablement sur une carte grand public actuelle.

Deux bugs résiduels du fichier canonique `ingestion.py` ne sont pas corrigés à la source par choix méthodologique : la méthode `BoundingBox.ymax` qui calcule `max(self.x1, self.y2, self.y3, self.y4)` au lieu de `self.y1` (sans impact sur le pipeline car cette propriété n'est pas utilisée en aval), et la méthode `OCREngine._load` qui référence un attribut `self._model` inexistant (entièrement contournée par notre wrapper qui ne passe jamais par cette branche).

### 10.2 Limites de conformité

La conformité à la Loi 25 couverte par ce projet est exclusivement *technique* : nous garantissons par construction qu'aucune donnée du document ne quitte la machine de l'utilisateur. Cependant, la Loi 25 exige aussi des éléments organisationnels que le code ne peut pas instrumenter à lui seul : la désignation d'un responsable de la protection des renseignements personnels, le registre des incidents, l'évaluation des facteurs relatifs à la vie privée (ÉFVP) pour les traitements à risque élevé, et la mise en place de mécanismes de consentement. Une organisation qui déploierait ce système devrait compléter avec ces processus.

### 10.3 Perspectives d'évolution

Trois pistes d'évolution nous paraissent prioritaires. Premièrement, le support des PDF natifs avec texte sélectionnable, qui éviterait l'OCR dans la majorité des cas administratifs (lettres, rapports). Deuxièmement, l'évaluation systématique sur un jeu de questions-réponses étiqueté, qui produirait des métriques de précision-rappel-citation au-delà des exemples qualitatifs présentés en section 9. Troisièmement, l'ajout d'une journalisation des requêtes côté utilisateur — non pas pour la télémétrie, qui contredirait nos engagements de confidentialité, mais pour permettre à l'utilisateur de revoir et corriger son propre historique localement.

---

## 11. Conclusion

Ce projet démontre qu'il est possible, avec les briques open-source disponibles en 2026, de construire un système de questions-réponses sur documents qui soit entièrement local, performant pour un usage administratif, et conforme aux contraintes de la Loi 25 du Québec. Le choix d'un backend unique pour la génération (Ollama), de deux pipelines spécialisés en parallèle (textuel via MPNet et FAISS, multimodal via ColQwen2 et Qdrant), et d'une couche d'enrobage qui respecte l'intégrité des contributions individuelles, nous a permis d'atteindre ce résultat sans réécriture massive du code des coéquipiers et avec une suite de tests rapide à exécuter.

Les principales leçons que nous retenons sont d'ordre méthodologique autant que technique. La discipline du typage strict dans Typer, la classification par contenu plutôt que par localisation, le dédoublonnage systématique des résultats de recherche, la détection multi-niveaux des binaires système : aucun de ces points n'est conceptuellement nouveau, mais chacun a été l'origine d'un bug en production, et chacun fait maintenant l'objet d'un test dédié. Cette accumulation de garde-fous, plus que la sophistication des modèles utilisés, est probablement ce qui distingue un prototype de cours d'un système qu'on pourrait responsabilité confier à un utilisateur final.

---

## Références

[1] Lewis P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.

[2] Faysse M. et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models*. arXiv:2407.01449.

[3] Reimers N. et Gurevych I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.

[4] Johnson J., Douze M. et Jégou H. (2017). *Billion-scale similarity search with GPUs*. arXiv:1702.08734 (FAISS).

[5] Huang Z. et al. (2019). *ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction*. ICDAR 2019 (corpus SROIE 2019).

[6] Jiang A. et al. (2023). *Mistral 7B*. arXiv:2310.06825.

[7] Qwen Team (2025). *Qwen2.5-VL Technical Report*. arXiv:2502.13923.

[8] Smith R. (2007). *An Overview of the Tesseract OCR Engine*. ICDAR 2007.

[9] Gouvernement du Québec. *Loi modernisant des dispositions législatives en matière de protection des renseignements personnels* (chapitre 25 des lois de 2021), dite « Loi 25 ».

[10] Ollama Project. <https://ollama.com> — consulté en mai 2026.

[11] Qdrant Project. <https://qdrant.tech> — consulté en mai 2026.

---

## Annexe — Reproduction des résultats

L'installation complète se fait en quatre commandes depuis un dépôt fraîchement extrait :

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull mistral:7b-instruct
ollama pull qwen2.5vl:7b
```

Sur Fedora 44 ou Debian/Ubuntu, ajouter Tesseract :

```bash
sudo dnf install tesseract tesseract-langpack-fra tesseract-langpack-eng
# OU :  sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
```

Vérifier que la suite de tests passe :

```bash
pytest
# 63 passed in 2.65s
```

Ingestion et requête du Pipeline 1 sur SROIE :

```bash
python scripts/phase1.py ingest --dir data/SROIE2019
python scripts/phase1.py db --question "What is the total amount?"
```

Lancement de l'interface graphique :

```bash
streamlit run app.py
```
