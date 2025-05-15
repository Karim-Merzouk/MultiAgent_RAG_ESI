import httpx
import google.generativeai as genai 
import pathlib
from IPython.display import Markdown, display
from pprint import pprint
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # Import the GROQ chat model
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic.v1 import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader, DirectoryLoader, UnstructuredMarkdownLoader
import bs4
import os
# Ensure these are at the file level for easier importing
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Document loading
loader = DirectoryLoader(
    path="dd",  
    loader_cls=UnstructuredMarkdownLoader,
    use_multithreading=True
)

txt = loader.load()

# Add this before your FAISS imports


class VectorStoreConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        # ----- Markdown -----
        r"\n# ", r"\n## ", r"\n### ", r"\n#### ",
        r"\n```\n",
        r"\n\*\*\*\n", r"\n---\n",

        # ----- LaTeX/Math -----
        r"\n\\begin{equation}",
        r"\n\\begin{align}",
        r"\n\\section{", r"\n\\subsection{",
        r"\n\$\$", r"\n\\\[",

        # ----- C Code -----
        r"\n}\n",
        r"\n// -{4,}\n",
        r"\n#ifdef ", r"\n#endif",
        r"\n}\n\n",

        # ----- Assembly -----
        r"\n\.section\s",
        r"\n\.global\s",
        r"\n[a-zA-Z_]+:\n",
        r"\n; -{4,}\n",

        # ----- Generic Fallbacks -----
        r"\n\n", r"\n", r" ", r""
    ],
    is_separator_regex=True,
    keep_separator=True,
    strip_whitespace=True,
)
fdocs = text_splitter.split_documents(txt)

# Embedding with CPU optimization
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "batch_size": 64,
        "normalize_embeddings": True
    }
)

# Create FAISS index
db = FAISS.from_documents(
    documents=fdocs,
    embedding=embeddings,
    normalize_L2=True
)


# To load it back later:
# loaded_db = FAISS.load_local(save_directory, embeddings)

# Now let's modify to use GROQ API instead of Ollama

# Set your GROQ API key - make sure to keep this secure!
# Load API key from .env file

# Load environment variables from .env file
load_dotenv()

# Get the GROQ API key from environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the GROQ LLM - replaced Ollama with GROQ
llm = ChatGroq(
    # model="llama3-70b-8192",  # You can choose "mixtral-8x7b-32768" or other models
    # You can choose "mixtral-8x7b-32768" or other models
    model='llama-3.3-70b-versatile',
    temperature=0,
    max_tokens=32768,  # Control response length
)

# The prompt template remains the same
prompt = ChatPromptTemplate.from_template(r'''[Contexte] {context}
[Question] {input}
# Syntaxe Détaillée du Langage Z

## 1. Introduction et Objectifs du Langage Z



math :
you're an expert at math so solve each question
use mathimathical symbols with their real size (big intégral symbols,
respect fractions size and nominator and dominator sizes correc their sizes )
------------------------------------
assambly:

- Architecture : 8086 (16-bit uniquement, **pas de registres 32-bit**).  
- Syntaxe : Compatible TASM/MASM avec segments explicites (DATA, STACK, CODE).  
- Exigences : Code 100% fonctionnel sous DOS, testé avec Turbo Assembler.  

[Instructions]  
1. **Priorité aux détails techniques** :  
   - Structurez le code en segments : DATA (déclarations), STACK (allocation), CODE (logique).  
   - Utilisez **uniquement des registres 16-bit** (AX, BX, CX, DX, etc.).  

2. **Réponse complète** :  
   - Incluez :  
     - L’initialisation manuelle des segments (DS, SS).  
     - Les interruptions DOS validées (INT 21h).  
   - Ajoutez une section "Pour compiler/exécuter" avec les commandes TASM/TLINK.  



[Exigences strictes]  
- **Interdits** : Registres 32-bit (EAX, EBX…), instructions 386+, BIOS.  
- **Validation** : Le code doit être testable sous DOSBox avec Turbo Assembler.  
- **Clarté** : Évitez les macros ou optimisations obscures.  

[Format de réponse attendu]  
```asm  
;=================================  
; [Nom du programme]  
;=================================  
DATA SEGMENT  
    ; Déclarations  
DATA ENDS  

STACK SEGMENT 
    ; Allocation de la pile  
STACK ENDS  

CODE SEGMENT  
    ASSUME CS:CODE, DS:DATA, SS:STACK  
    start:  
        ; Initialisation DS 
        ; Logique du programme  
CODE ENDS  
END start  
--DEclarer procedures before "CODE ENDS"
//////////////












//////////////////






Le langage Z est un langage algorithmique conçu principalement à des fins pédagogiques. Son objectif est d'enseigner l'algorithmique et les structures de données en permettant aux utilisateurs d'écrire des algorithmes abstraits. Ces algorithmes sont basés sur des modèles de machines abstraites, ce qui permet de se concentrer sur la logique de résolution de problèmes avant de passer à une implémentation dans un langage de programmation spécifique comme Pascal ou C.

**Caractéristiques Clés pour l'IA :**
* Abstraction de la représentation mémoire.
* Syntaxe claire et structurée, facilitant l'analyse et la génération.
* Support des structures de données fondamentales.
* Modularité via les actions et les fonctions.

## 2. Structure Générale d'un Z-Algorithme

Un Z-algorithme est organisé en modules. Le premier module est toujours le module principal. Les modules suivants peuvent être des actions (procédures) ou des fonctions.

```Z
ALGORITHME Nom_Algorithme  // Optionnel, mais bon pour la clarté

SOIENT
    // Section des déclarations globales
    // Déclaration des constantes globales
    // Déclaration des types globaux
    // Déclaration des variables globales
    // Annonces des modules (actions et fonctions) qui seront définis plus bas

DEBUT // Début du module principal
    // Instructions du module principal
FIN // Fin du module principal

// Définition des modules (Actions et Fonctions) annoncés
ACTION Nom_Action_1 (paramètres)
    SOIENT
        // Déclarations locales à l'action
    DEBUT
        // Instructions de l'action
    FIN

FONCTION Nom_Fonction_1 (paramètres) : Type_Retour
    SOIENT
        // Déclarations locales à la fonction
    DEBUT
        // Instructions de la fonction
        Nom_Fonction_1 := Valeur_De_Retour // Instruction de retour
    FIN

// ... autres actions et fonctions
```

**Points importants pour l'IA :**
* Identifier la délimitation claire entre la section `SOIENT` (déclarations) et `DEBUT`/`FIN` (bloc d'instructions).
* Reconnaître le module principal comme point d'entrée.
* Comprendre la portée des variables (globales vs. locales à un module).
* Les annonces de modules dans `SOIENT` avant leur définition complète.

## 3. Commentaires

Les commentaires sont essentiels pour la lisibilité et peuvent être ignorés par l'interpréteur ou le compilateur.


**Points importants pour l'IA :**
* Le modèle doit apprendre à identifier et potentiellement ignorer le contenu des commentaires lors de l'analyse de la logique du code.
* Lors de la génération de code, le modèle pourrait être entraîné à insérer des commentaires pertinents.

## 4. Identificateurs

Les identificateurs sont les noms donnés aux variables, constantes, types, actions, fonctions, etc.
* Doivent commencer par une lettre.
* Peuvent contenir des lettres, des chiffres et le caractère souligné (`_`).
* Ne sont pas sensibles à la casse (par convention, mais à vérifier si les outils fournis le sont).
* Ne doivent pas être des mots-clés réservés du langage.

**Exemples :**
`Compteur`, `Total_Somme`, `Est_Valide`, `Ma_Procedure`

**Points importants pour l'IA :**
* Apprendre les règles de formation des identificateurs valides.
* Distinguer les identificateurs des mots-clés.

## 5. Mots-Clés Réservés

Le langage Z possède un ensemble de mots-clés qui ont une signification spéciale et ne peuvent pas être utilisés comme identificateurs.
Voici une liste non exhaustive (à compléter à partir de l'index des mots-clés du document `Khawarizm_.pdf`) :

`ACTION`, `ALGORITHME`, `APPEL`, `ARB`, `ARM`, `BOOLEEN`, `BOOLEENS`, `CAR`, `CARACTERE`, `CARACTERES`, `CHAINE`, `CHAINES`, `CONST`, `CREERFILE`, `CREERNOEUD`, `CREERPILE`, `CREER_ARB`, `CREER_ARM`, `CREER_FILE`, `CREER_LISTE`, `DE`, `DEBUT`, `DEFILER`, `DEPILER`, `DES`, `ECRIRE`, `ELEMENT`, `EMPILER`, `ENFILER`, `ENTIER`, `ENTIERS`, `ET`, `FAUX`, `FICHIER`, `FILE`, `FIN`, `FINSI`, `FINTANTQUE`, `FINPOUR`, `FONCTION`, `LIRE`, `LISTE`, `MOD`, `NIL`, `NON`, `OU`, `PILE`, `POUR`, `PROCEDURE` (synonyme d'ACTION), `REEL` (à vérifier si supporté, les documents mentionnent principalement ENTIER), `REPETER`, `RETOURNE` (pourrait être utilisé avec FONCTION), `SI`, `SINON`, `SOIT`, `SOIENT`, `STRUCT`, `TABLEAU`, `TANTQUE`, `TYPE`, `UN`, `UNE`, `VECTEUR`, `VRAI`.

**Points importants pour l'IA :**
* Mémoriser cette liste pour éviter les conflits lors de la génération de noms.
* Utiliser ces mots-clés pour comprendre la structure et la sémantique du code.

## 6. Types de Données

### 6.1. Types Scalaires Standards

* **ENTIER**: Pour les nombres entiers (positifs, négatifs ou nuls).
    * Exemple de déclaration : `Age : ENTIER`
* **BOOLEEN**: Pour les valeurs logiques.
    * Valeurs possibles : `VRAI`, `FAUX`.
    * Exemple de déclaration : `Trouve : BOOLEEN`
* **CAR**: Pour un caractère unique.
    * Les littéraux caractères sont souvent entourés d'apostrophes (ex: `'A'`).
    * Exemple de déclaration : `Lettre : CAR`
* **CHAINE**: Pour les séquences de caractères.
    * Les littéraux chaînes sont souvent entourés de guillemets (ex: `"Bonjour"`).
    * Exemple de déclaration : `Message : CHAINE`

### 6.2. Types Composés

Le langage Z permet de définir des types plus complexes :

* **Tableaux / Vecteurs**: Collections d'éléments de même type, accessibles par un ou plusieurs indices.
    * Syntaxe de déclaration : `Mon_Tableau : TABLEAU [borne_inf..borne_sup] DE Type_Element`
    * Exemple : `Notes : TABLEAU [1..20] DE ENTIER`
    * Les vecteurs peuvent être statiques ou dynamiques (`ALLOC_TAB`, `LIBER_TAB`).

* **Structures (Enregistrements)**: Collections d'éléments de types potentiellement différents, appelés champs, accessibles par leur nom.
    * Syntaxe de déclaration de type :
        ```Z
        TYPE
            Nom_Type_Structure = STRUCTURE
                Champ1 : Type1 ;
                Champ2 : Type2 ;
                // ...
            FINSTRUCTURE // ou FIN STRUCT
        ```
    * Déclaration de variable : `Ma_Var : Nom_Type_Structure`
    * Exemple :
        ```Z
        TYPE
            Personne = STRUCTURE
                Nom : CHAINE[30] ; // Chaîne de longueur fixe
                Age : ENTIER ;
            FINSTRUCTURE
        SOIENT
            Etudiant : Personne ;
        ```
    * Les structures peuvent être statiques ou dynamiques (`ALLOC_STRUCT`, `LIBER_STRUCT`).

* **Pointeurs**: Utilisés pour les structures de données dynamiques (listes, arbres). Un pointeur contient l'adresse d'une variable.
    * Syntaxe de déclaration : `Ptr_Var : ^Type_Cible` (la notation exacte peut varier, se référer aux exemples des PDFs).
    * Valeur spéciale : `NIL` (indique que le pointeur ne pointe sur rien).

* **Listes Linéaires Chaînées**: Séquences d'éléments (nœuds) où chaque nœud contient une donnée et un pointeur vers le nœud suivant (et précédent pour les listes bilatérales).
    * Implique la définition d'un type nœud (structure) et l'utilisation de pointeurs.
    * Opérations typiques : `CREER_LISTE`, `INSERER`, `SUPPRIMER`, `CHERCHER`.

* **Files d'attente (Queues)**: Structure de type FIFO (Premier Entré, Premier Sorti).
    * Opérations typiques : `CREER_FILE`, `ENFILER`, `DEFILER`, `FILE_VIDE`.

* **Piles (Stacks)**: Structure de type LIFO (Dernier Entré, Premier Sorti).
    * Opérations typiques : `CREER_PILE`, `EMPILER`, `DEPILER`, `PILE_VIDE`.

* **Arbres (Binaires, M-aires)**: Structures de données hiérarchiques.
    * Implique la définition d'un type nœud.
    * Opérations typiques : `CREER_ARB`, `CREERNOEUD`, `FG` (fils gauche), `FD` (fils droit), `PERE`.

* **Fichiers**: Pour la persistance des données.
    * Déclaration : `Mon_Fichier : FICHIER DE Type_Element`
    * Opérations : `OUVRIR`, `FERMER`, `LIRESEQ`, `ECRIRESEQ`.

* **Types Composés Complexes**: Le langage Z permet des imbrications comme `PILE DE FILES DE LISTES DE ENTIER`.
    * Exemple : `MaStructureComplexe : PILE DE LISTE DE CAR`

**Points importants pour l'IA :**
* Reconnaître les mots-clés de chaque type.
* Comprendre la syntaxe de déclaration pour chaque type (dimensions pour les tableaux, champs pour les structures, etc.).
* Associer les opérations de haut niveau (ex: `EMPILER`) avec le type de données correspondant (ex: `PILE`).

## 7. Déclarations

Toutes les entités (constantes, types, variables, modules) doivent être déclarées avant leur utilisation. Les déclarations se font principalement dans la section `SOIENT`.

### 7.1. Déclaration de Constantes

Permet de donner un nom à une valeur fixe.
Syntaxe : `CONST Nom_Constante = Valeur ;`
Exemple : `CONST PI = 3.14159 ;`
Exemple : `CONST MAX_TAILLE = 100 ;`

### 7.2. Déclaration de Types

Permet de définir de nouveaux types de données (surtout pour les structures, tableaux, etc.).
Syntaxe :
```Z
TYPE
    Nouveau_Nom_Type = Définition_Type ;
    // ... autres définitions de type
```
Exemple (déjà vu pour les structures) :
```Z
TYPE
    Point = STRUCTURE
        X, Y : ENTIER ;
    FINSTRUCTURE
```

### 7.3. Déclaration de Variables

Associe un nom à un emplacement mémoire qui peut stocker une valeur d'un type donné.
Syntaxe générale : `[SOIT | SOIENT] Liste_Identificateurs <Séparateur> Type_Donnée ;`
* `Liste_Identificateurs`: Un ou plusieurs noms de variables, séparés par des virgules.
* `<Séparateur>`: Peut être `:`, `UN`, `UNE`, `DES`. Le choix est souvent stylistique ou pour la lisibilité.
* `Type_Donnée`: Un type standard ou un type préalablement défini.

Exemples :
`SOIENT I, J, K : ENTIER ;`
`SOIT Nom_Utilisateur UNE CHAINE ;`
`SOIENT Scores DES TABLEAU [1..10] DE ENTIER ;`
`SOIT P1 UN Point ;`

### 7.4. Annonce de Modules (Actions et Fonctions)

Avant de définir une action ou une fonction (si elle n'est pas définie avant son premier appel, ce qui est la norme pour les définitions après le module principal), elle doit être annoncée dans la section `SOIENT` du module qui l'appelle ou globalement.
Syntaxe Annonce Action : `ACTION Nom_Action (Liste_Paramètres_Formels_Avec_Types) ;`
Syntaxe Annonce Fonction : `FONCTION Nom_Fonction (Liste_Paramètres_Formels_Avec_Types) : Type_Retour ;`

Exemple :
```Z
SOIENT
    // ... autres déclarations
    ACTION Afficher_Message (Msg : CHAINE) ;
    FONCTION Calculer_Somme (A, B : ENTIER) : ENTIER ;
```

**Points importants pour l'IA :**
* Identifier le type de déclaration (constante, type, variable, module).
* Extraire le nom, le type et la valeur (pour les constantes) ou la structure (pour les types).
* Comprendre la portée des déclarations (globale si dans `SOIENT` du module principal, locale sinon).

## 8. Expressions

Une expression est une combinaison de valeurs (littéraux, constantes, variables), d'opérateurs et d'appels de fonctions, qui s'évalue en une unique valeur.

### 8.1. Expressions Arithmétiques
Opérateurs : `+` (addition), `-` (soustraction, unaire moins), `*` (multiplication), `/` (division réelle), `DIV` (division entière), `MOD` (modulo).
Priorité des opérateurs : `*`, `/`, `DIV`, `MOD` ont une priorité plus élevée que `+`, `-`. Les parenthèses `()` peuvent être utilisées pour forcer l'ordre d'évaluation.
Exemple : `(A + B) * C / 2`

### 8.2. Expressions Logiques (Booléennes)
Opérateurs : `ET` (ET logique), `OU` (OU logique), `NON` (NON logique).
Valeurs : `VRAI`, `FAUX`.
Exemple : `(Age >= 18) ET (Est_Etudiant OU A_Reduction)`
Exemple : `NON Trouve`

### 8.3. Expressions Relationnelles (de Comparaison)
Opérateurs : `=`, `<>` (ou `#` pour différent), `<`, `<=`, `>`, `>=`.
Résultat : Toujours une valeur booléenne (`VRAI` ou `FAUX`).
Exemple : `X > Y`, `Nom = "Test"`

### 8.4. Expressions sur Chaînes de Caractères
Opérateur : `+` (concaténation).
Exemple : `"Bonjour" + " " + Nom_Utilisateur`

### 8.5. Accès aux Éléments de Types Composés
* Tableaux : `Nom_Tableau[Index]`
    Exemple : `Notes[I]`
* Structures : `Nom_Variable_Structure.Nom_Champ`
    Exemple : `Etudiant.Age`
* Pointeurs (déréférencement) : `Nom_Pointeur^` (pour accéder à la valeur pointée) ou `Nom_Pointeur^.Champ` si le pointeur pointe sur une structure. La notation exacte peut varier (ex: `INFO(P)` pour le contenu d'un nœud pointé par P dans une liste). Se référer aux opérations spécifiques des structures de données dans les PDFs.

### 8.6. Appels de Fonctions
Une fonction, lorsqu'elle est appelée dans une expression, retourne une valeur qui est utilisée dans le calcul de l'expression.
Syntaxe : `Nom_Fonction (Paramètre_Actuel_1, Paramètre_Actuel_2, ...)`
Exemple : `Resultat := Calculer_Somme(N1, N2) + 10`

**Points importants pour l'IA :**
* Parser correctement les expressions en respectant la priorité des opérateurs.
* Identifier le type résultant d'une expression.
* Gérer les appels de fonction et l'accès aux éléments de structures de données.

## 9. Instructions (Actions Élémentaires et Structures de Contrôle)

Les instructions décrivent les opérations à effectuer.

### 9.1. Affectation
Attribue la valeur d'une expression à une variable.
Syntaxe : `Nom_Variable := Expression ;`
Le type de `Expression` doit être compatible avec le type de `Nom_Variable`.
Exemples :
`Compteur := Compteur + 1 ;`
`Message_Bienvenue := "Bonjour !" ;`
`Est_Majeur := Age >= 18 ;`
`Mon_Tableau[Indice] := Valeur ;`
`Mon_Record.Champ := Autre_Valeur ;`

L'affectation globale est permise pour les types composés (ex: copier une structure entière dans une autre de même type).

### 9.2. Lecture (Entrée)
Permet d'obtenir des données depuis une source d'entrée standard (généralement le clavier) et de les stocker dans des variables.
Syntaxe : `LIRE ( Variable_1, Variable_2, ... ) ;`
Exemple : `LIRE ( Age, Nom ) ;`

### 9.3. Écriture (Sortie)
Permet d'afficher les valeurs d'expressions ou des messages sur une sortie standard (généralement l'écran).
Syntaxe : `ECRIRE ( Expression_1, Expression_2, ... ) ;`
Les expressions peuvent être des littéraux, des variables, ou des calculs plus complexes.
Exemple : `ECRIRE ( "Le résultat est : ", Somme / Nombre_Elements ) ;`
Exemple : `ECRIRE ( "Bonjour ", Nom_Utilisateur, " !" ) ;`

### 9.4. Structures de Contrôle

#### 9.4.1. Conditionnelle (SI ... SINON ... FINSI)
Exécute des blocs d'instructions en fonction de la valeur d'une condition booléenne.
* Forme simple :
    ```Z
    SI Condition_Booleenne [ALORS | :] // ALORS ou : est optionnel, : est fréquent
        // Bloc d'instructions si la condition est VRAI
    FINSI
    ```
* Forme avec alternative :
    ```Z
    SI Condition_Booleenne [ALORS | :]
        // Bloc d'instructions si la condition est VRAI
    SINON
        // Bloc d'instructions si la condition est FAUX
    FINSI
    ```
* Imbrication :
    ```Z
    SI Condition_1 :
        // ...
    SINON
        SI Condition_2 :
            // ...
        SINON
            // ...
        FINSI
    FINSI
    ```
Exemple :
```Z
SI Note >= 10 :
    ECRIRE ( "Admis" ) ;
SINON
    ECRIRE ( "Ajourné" ) ;
FINSI
```

#### 9.4.2. Boucle TANTQUE (Tant que ... Faire)
Répète un bloc d'instructions tant qu'une condition booléenne reste vraie. La condition est testée avant chaque itération.
Syntaxe :
```Z
TANTQUE Condition_Booleenne [FAIRE | :] // FAIRE ou : est optionnel
    // Bloc d'instructions à répéter
FINTANTQUE // ou FTQ
```
Exemple :
```Z
SOIENT I : ENTIER ;
I := 1 ;
TANTQUE I <= 10 :
    ECRIRE ( I ) ;
    I := I + 1 ;
FINTANTQUE
```

#### 9.4.3. Boucle POUR (Pour ... De ... À ...)
Répète un bloc d'instructions un nombre déterminé de fois, en utilisant une variable compteur.
Syntaxe :
```Z
POUR Variable_Compteur := Valeur_Initiale , Valeur_Finale [, Pas] [FAIRE | :] // Pas est optionnel, défaut 1
    // Bloc d'instructions à répéter
FINPOUR // ou FPOUR
```
* `Variable_Compteur` : Doit être d'un type ordinal (généralement ENTIER).
* `Valeur_Initiale`, `Valeur_Finale` : Expressions évaluant au type du compteur.
* `Pas` (optionnel) : Expression entière. Si positif, le compteur est incrémenté. Si négatif, il est décrémenté. Si omis, le pas est de `1`.

Exemple (incrémentation) :
```Z
POUR I := 1 , 5 : // Pas de 1 par défaut
    ECRIRE ( "Itération numéro : ", I ) ;
FINPOUR
```
Exemple (décrémentation avec pas explicite) :
```Z
POUR J := 10 , 1 , -2 :
    ECRIRE ( J ) ;
FINPOUR // Affichera 10, 8, 6, 4, 2
```

#### 9.4.4. Boucle REPETER ... JUSQU'A (À vérifier si présent dans les documents Z fournis)
Répète un bloc d'instructions jusqu'à ce qu'une condition devienne vraie. Le bloc est exécuté au moins une fois car la condition est testée après l'itération.
Syntaxe (typique, à confirmer pour Z) :
```Z
REPETER
    // Bloc d'instructions
JUSQU'A Condition_Booleenne
```
*Note : Les documents fournis se concentrent sur `TANTQUE` et `POUR`. Si `REPETER...JUSQU'A` est supporté, sa syntaxe exacte doit être vérifiée.*

**Points importants pour l'IA :**
* Identifier chaque type d'instruction.
* Pour les affectations, valider la compatibilité des types.
* Pour les structures de contrôle, identifier la condition et les blocs d'instructions associés.
* Comprendre la sémantique de chaque boucle (condition de test, modification du compteur).

## 10. Modules : Actions et Fonctions

Les modules permettent de structurer le code, de le réutiliser et de le rendre plus lisible.

### 10.1. Actions (Procédures)
Un bloc d'instructions nommé qui peut être appelé pour effectuer une tâche spécifique. Ne retourne pas de valeur directement (mais peut modifier des variables passées en paramètre ou des variables globales).

* **Annonce (Déclaration anticipée)**: (Vue dans la section Déclarations)
    `ACTION Nom_Action (P1:Type1 ; P2:Type2 ; ... ) ;`
    Les paramètres dans l'annonce incluent leurs types.

* **Définition**:
    ```Z
    ACTION Nom_Action (P_formel_1 [:Type1] ; P_formel_2 [:Type2] ; ... ) // Types optionnels ici si déjà dans l'annonce, mais bon pour la clarté
        SOIENT
            // Déclarations locales (variables, constantes, types spécifiques à l'action)
        DEBUT
            // Corps de l'action (instructions)
        FIN
    ```
    Les paramètres formels (`P_formel_1`, etc.) sont des placeholders pour les valeurs qui seront passées lors de l'appel.
    Le passage de paramètres en Z est **par référence** (ou adresse) par défaut. Cela signifie que si l'action modifie un paramètre, la variable originale passée lors de l'appel est modifiée. (Il est important de vérifier s'il existe un mécanisme de passage par valeur explicitement).

* **Appel**:
    `APPEL Nom_Action (Param_Actuel_1, Param_Actuel_2, ... ) ;`
    Ou plus simplement : `Nom_Action (Param_Actuel_1, Param_Actuel_2, ... ) ;` (L'utilisation de `APPEL` est souvent une convention plus ancienne ou pour la clarté).
    Les paramètres actuels sont les valeurs ou variables réelles passées à l'action.

Exemple :
```Z
SOIENT
    ACTION Afficher_Somme (A:ENTIER ; B:ENTIER) ; // Annonce

// ... dans le module principal ...
DEBUT
    // ...
    APPEL Afficher_Somme (Nombre1, Nombre2) ;
    // ...
FIN

// Définition de l'action
ACTION Afficher_Somme (A:ENTIER ; B:ENTIER)
    SOIENT
        Somme_Locale : ENTIER ;
    DEBUT
        Somme_Locale := A + B ;
        ECRIRE ( "La somme est : ", Somme_Locale ) ;
    FIN
```

### 10.2. Fonctions
Similaires aux actions, mais elles **retournent une valeur** d'un type spécifié.

* **Annonce (Déclaration anticipée)**: (Vue dans la section Déclarations)
    `FONCTION Nom_Fonction (P1:Type1 ; P2:Type2 ; ... ) : Type_Retour ;`

* **Définition**:
    ```Z
    FONCTION Nom_Fonction (P_formel_1 [:Type1] ; P_formel_2 [:Type2] ; ... ) : Type_Retour
        SOIENT
            // Déclarations locales
        DEBUT
            // Corps de la fonction (instructions)
            // ...
            Nom_Fonction := Expression_De_Retour ; // Instruction cruciale pour retourner la valeur
            // ...
        FIN
    ```
    L'instruction `Nom_Fonction := Expression_De_Retour ;` assigne la valeur à retourner au nom de la fonction elle-même. C'est ainsi que la valeur est renvoyée à l'appelant.
    Le passage de paramètres est également par référence par défaut.

* **Appel**:
    Une fonction est appelée dans une expression, là où une valeur de son type de retour est attendue.
    `Variable_Resultat := Nom_Fonction (Param_Actuel_1, Param_Actuel_2, ... ) ;`
    `ECRIRE ( "Résultat : ", Nom_Fonction(X, Y) ) ;`

Exemple :
```Z
SOIENT
    FONCTION Calculer_Produit (Val1:ENTIER ; Val2:ENTIER) : ENTIER ; // Annonce

// ... dans le module principal ...
DEBUT
    SOIENT Resultat, N1, N2 : ENTIER ;
    LIRE(N1, N2);
    Resultat := Calculer_Produit (N1, N2) ;
    ECRIRE ( "Le produit est : ", Resultat ) ;
FIN

// Définition de la fonction
FONCTION Calculer_Produit (Val1:ENTIER ; Val2:ENTIER) : ENTIER
    SOIENT
        Produit_Local : ENTIER ;
    DEBUT
        Produit_Local := Val1 * Val2 ;
        Calculer_Produit := Produit_Local ; // Retour de la valeur
    FIN
```

### 10.3. Récursivité
Le langage Z supporte les modules récursifs, c'est-à-dire qu'une action ou une fonction peut s'appeler elle-même.

**Points importants pour l'IA :**
* Distinguer clairement les actions des fonctions (retour de valeur).
* Comprendre la syntaxe d'annonce, de définition et d'appel.
* Analyser les listes de paramètres (formels et actuels) et leurs types.
* Identifier l'instruction de retour dans les fonctions.
* Comprendre le mécanisme de passage de paramètres (par référence).
* Détecter les appels récursifs.

## 11. Fonctions Prédéfinies et Opérations de Haut Niveau

Le langage Z fournit des fonctions et opérations intégrées pour des tâches courantes.

### 11.1. Fonctions Mathématiques Standards
* `MOD (Dividende, Diviseur)`: Reste de la division entière.
* `MAX (Valeur1, Valeur2, ...)`: Retourne la plus grande des valeurs.
* `MIN (Valeur1, Valeur2, ...)`: Retourne la plus petite des valeurs.
* `EXP (Base, Exposant)`: Exponentiation (Base élevée à la puissance Exposant). (À vérifier si c'est `PUIS` ou `EXP` selon les documents).
* `ABS (Nombre)`: Valeur absolue. (À vérifier)
* `SQRT (Nombre)`: Racine carrée. (À vérifier)

### 11.2. Fonctions de Génération Aléatoire
* `ALEACHAINE (Longueur)`: Génère une chaîne de caractères aléatoire de la longueur spécifiée.
* `ALEANOMBRE (Borne_Inf, Borne_Sup)` ou `ALEAENTIER`: Génère un nombre entier aléatoire dans un intervalle. (La syntaxe exacte peut varier).

### 11.3. Fonctions sur Chaînes de Caractères
* `LONGCHAINE (Chaine)`: Retourne la longueur de la chaîne.
* `CARACT (Chaine, Position)`: Retourne le caractère à une position donnée dans la chaîne.
* `SOUSCHAINE (Chaine, Debut, Longueur)`: Extrait une sous-chaîne. (À vérifier)
* `CONCAT (Chaine1, Chaine2)`: Concatène deux chaînes (équivalent à l'opérateur `+`). (À vérifier)

### 11.4. Opérations de Haut Niveau sur Structures de Données
Le langage Z est doté d'opérations abstraites pour manipuler les structures de données, masquant les détails d'implémentation.
Exemples (liste non exhaustive, à compléter avec l'index des mots-clés de `Khawarizm_.pdf`) :

* **Vecteurs**:
    * `ELEMENT (Vecteur, Indice)`: Accéder à un élément.
    * `AFF_ELEMENT (Vecteur, Indice, Valeur)`: Modifier un élément.
    * `ALLOC_TAB (Nom_Tableau_Dynamique, Dimensions)`: Allouer dynamiquement un tableau.
    * `LIBER_TAB (Nom_Tableau_Dynamique)`: Libérer la mémoire.

* **Structures**:
    * `STRUCT (Variable_Structure, Nom_Champ)`: Accéder à un champ (moins courant que la notation `.` ).
    * `AFF_STRUCT (Variable_Structure, Nom_Champ, Valeur)`: Modifier un champ.
    * `ALLOC_STRUCT (Nom_Variable_Structure_Dynamique)`: Allouer dynamiquement.
    * `LIBER_STRUCT (Nom_Variable_Structure_Dynamique)`: Libérer.

* **Listes**:
    * `CREER_LISTE (Liste)`
    * `INSERER_TETE (Liste, Valeur)`
    * `SUPPRIMER_TETE (Liste)`
    * `LISTE_VIDE (Liste)`
    * Opérations spécifiques pour listes bilatérales (`CREER_LISTEBI`).

* **Files**:
    * `CREER_FILE (File)`
    * `ENFILER (File, Valeur)`
    * `DEFILER (File, Variable_Pour_Valeur_Defilee)`
    * `FILE_VIDE (File)`

* **Piles**:
    * `CREER_PILE (Pile)`
    * `EMPILER (Pile, Valeur)`
    * `DEPILER (Pile, Variable_Pour_Valeur_Depilee)`
    * `PILE_VIDE (Pile)`

* **Arbres**:
    * `CREER_ARB (Arbre)`
    * `CREERNOEUD (Valeur)`: Crée un nœud isolé.
    * `FG (Noeud)`: Accède au fils gauche.
    * `FD (Noeud)`: Accède au fils droit.
    * `PERE (Noeud)`: Accède au père.
    * `INFO (Noeud)`: Accède à l'information du nœud.
    * `AFF_FG (Noeud_Parent, Noeud_Fils_Gauche)`: Affecte un fils gauche.
    * `AFF_INFO (Noeud, Valeur)`: Modifie l'information du nœud.
    * `LIBERERNOEUD (Noeud)`

**Points importants pour l'IA :**
* Créer un catalogue de toutes les fonctions prédéfinies et opérations de haut niveau.
* Pour chacune, connaître son nom, ses paramètres (nombre et type), et ce qu'elle fait ou retourne.
* Comprendre que ces opérations abstraient la manipulation de bas niveau des pointeurs et de la mémoire pour les structures de données.

## 12. Allocation Dynamique

Pour les tableaux et les structures, le langage Z permet l'allocation dynamique de mémoire.
* `ALLOUER (Pointeur_Variable, Type_A_Allouer)` : Fonction générique d'allocation (si elle existe, sinon utiliser les spécifiques comme `ALLOC_TAB`, `ALLOC_STRUCT`, ou `CREERNOEUD`).
* `LIBERER (Pointeur_Variable)` : Fonction générique de libération.
* Les documents mentionnent `ALLOC_TAB`, `LIBER_TAB` pour les tableaux dynamiques et `ALLOC_STRUCT`, `LIBER_STRUCT` pour les structures dynamiques. Pour les listes/arbres, `CREERNOEUD` alloue un nouveau nœud et `LIBERERNOEUD` le libère.

**Points importants pour l'IA :**
* Identifier les opérations d'allocation et de libération.
* Comprendre que ces opérations sont nécessaires pour les structures de données dont la taille n'est pas connue à la compilation ou qui peuvent grandir/rétrécir pendant l'exécution.

## Conclusion pour l'Entraînement du Modèle d'IA

Pour entraîner efficacement un modèle d'IA sur le langage Z :
1.  **Corpus de Données**: Rassembler un grand nombre d'exemples de Z-algorithmes corrects et variés, couvrant toutes les constructions syntaxiques et les structures de données.
2.  **Tokenisation**: Définir un tokenizer capable de décomposer le code Z en unités significatives (mots-clés, identificateurs, opérateurs, littéraux).
3.  **Parsing et Arbre Syntaxique Abstrait (AST)**: Le modèle devrait idéalement apprendre à construire une représentation interne de la structure du code, similaire à un AST. Cela aide à comprendre les relations entre les différentes parties du code.
4.  **Apprentissage Séquentiel**: Pour la génération de code, des modèles séquence-à-séquence (comme les Transformers) sont bien adaptés.
5.  **Validation Sémantique**: Au-delà de la syntaxe, le modèle devrait (idéalement, à un stade avancé) apprendre des aspects sémantiques (ex: compatibilité des types, portée des variables).
6.  **Utilisation des Mots-Clés**: L'IA doit apprendre l'importance et l'usage de chaque mot-clé (`SI`, `TANTQUE`, `ACTION`, `ENTIER`, etc.).
7.  **Gestion des Structures de Données**: Comprendre comment déclarer et utiliser les opérations de haut niveau pour chaque structure de données est fondamental.

Ce guide syntaxique détaillé devrait fournir une base solide pour l'apprentissage de votre modèle IA. N'oubliez pas de vous référer extensivement aux exemples et à l'index des mots-clés dans les documents PDF fournis pour affiner la compréhension de chaque construction.








''')

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
retriever = db.as_retriever(
    search_kwargs={"k": 6}  # Retrieve top 4 most relevant documents
)


retriever_chain = create_retrieval_chain(retriever, document_chain)



# Configure API key for Gemini
genai.configure(api_key="AIzaSyD9MlsyGGWbNM40_92Z7kSJqCInldC7Owc")

def query_system(question):
    """Function to query the system with a question and get a response."""
    result = retriever_chain.invoke({'input': question})
    return result['answer']


# This conditional ensures that query_system is only run when try.py is executed directly
# and not when imported by streamlit_interface.py
if __name__ == "__main__":
    # Example query
    query = "donne un program assambleur pour donner les nombres premiers inférieur à un nombre entré N"
    response = query_system(query)
    groqrr = response
    
    # Configure API key (get yours from Google AI Studio)
    print('hiiiii')

    # Configure API key (get from https://aistudio.google.com/)
    # Replace with your actual key
    genai.configure(api_key="AIzaSyD9MlsyGGWbNM40_92Z7kSJqCInldC7Owc")

    # Path to your LOCAL PDF file
    # 🚨 Update this to your actual file path

    # Read PDF bytes from local file

    # Initialize the model
    # Check for latest model names
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

    # Generate response
    response = model.generate_content(
        [
            fr"""
-a chat bot was asked this quesion: {query}\n
-and the chabot gave me this response
{groqrr}\n
-and you have to do all this: 
if the response has fractions or integrals then keep their big sizes an keep gaps between expressions
correct the logic for equations, assambly program, algorithems, programs, methods, and check if program and script works, if the logic is false try to correct it
do anything to make it works, revise it and change anything to make it works
give the output as a markdown file
""",  # Your prompt,  # Your prompt
            # PDF data
        ]
    )

    print(response.text)
    print('finnn')
