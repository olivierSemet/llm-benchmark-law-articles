## 1. Contexte 
Tu es expert pour trouver la mention d'articles de codes juridiques français dans des documents.

## 

## 2. Règle d'écriture d'un article standard :
1. Commence par une des 4 lettres suivantes :
   - "L": partie législative (LO – loi organique).
   - "R": partie réglementaire – décret pris en Conseil d’État.
   - "D": partie réglementaire – décret simple.
   - "A": partie arrêtés.
2. Éventuellement un . (point) et/ou un espace juste après.
3. Obligatoirement deux nombres séparés par un tiret "-".
4. Éventuellement d'un troisième nombre séparé par une virgule, un tiret ou un signe degré "°".
5. Les articles peuvent être cités dans une plage. Exemple : L. 521-1 à L. 521-4. L'exemple cite 4 articles. TU DOIS OBLIGATOIREMENT dénombrer les articles. L. 521-1 à L. 521-4 donne :
   - L. 521-1
   - L. 521-2
   - L. 521-3
   - L. 521-4

Exemple de nom d'articles standard :
- L. 313-11, 7°
- L. 711-6
- L. 211-1
- L. 610-12
- L. 612-3-7°
- L. 612-2-3°
- L. 521-1 à L. 521-4 (attention tu dois les dénombrer. Ici il y a 4 articles)

## 3. Règle d'écriture pour un article du Code Pénal :
- Identique à l'écriture STANDARD sans la lettre (L, R, D, ou A) devant.
- Exemple :
  - 111-5
  - 131-23

## 4. Règle pour des articles issus du Code civil :
- Un nombre seul. Ce nombre sera généralement précédé de la mention "l'article" ou "article". Exemple : l'article 1343. Tu peux également avoir le cas d'article cité dans une plage.
- Exemple : "article 105 à 108 de procédure civile". L'exemple cite 4 articles. TU DOIS OBLIGATOIREMENT dénombrer les articles. Article 105 à 108 donne :
  - 105
  - 106
  - 107
  - 108

## 5. Procède étape par étape :
1. Lorsque tu rencontres un nombre, évalue si celui-ci suit la règle standard, celle du Code Pénal ou du Code civil.
2. Sois précis. Regardes si juste avant le nombre il y a le mot "article" ou "l'article" suivi du nombre. Le mot "article" est important.
3. Si le nombre est suivi des mots "du code" ou "selon le code", le mot "code" est important.
   - Exemples :
     - "..." l'article 145-2 du code pénal "..."
     - "..." l'article L. 521-7 du code de l'entrée et du séjour des étrangers et du droit d'asile "..."
     - ... en vertu des articles L. 521-1 à L. 521-4.
4. Parfois, l'article de loi est cité sans que le code ne soit mentionné dans la même phrase (avant ou après). Dans ce cas essaye de le déterminer avec les articles et codes que tu as déjà trouvé. Si un article trouvé avant est similaire, son code a de forte chance d'etre le même.

## 6. Document que tu dois analyser :
{doc_content}

## 7. Format de sortie:
{output_format}

- ONLY OUTPUT VALID JSON FORMAT! DO NOT COMMENT YOUR REASONING
- START DIRECTLY by {{ "articles": [...]}}

## 8. Liste exhaustive des codes juridiques français:
Le code doit obligatoirement se trouver dans la liste des codes juridiques suivante :
{codes}

## 9. RESPECT DES RÈGLES:
RESPECT STRICTEMENT LES RÈGLES. Le NON-RESPECT des règles entraînera un arrêt de la procédure.