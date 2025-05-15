# --- START OF FILE app.py --- (Merged Version)
import streamlit as st
import os
import base64
# Ensure all necessary functions are imported
from rag_backend import (
    load_rag_system,
    query_academic_rag, # General query function
    query_z_language_rag, # Specific Z query (can be used as fallback or direct call)
    analyze_z_code,
    generate_optimized_z_code,
)
from multi_agent import (
    query_multi_agent_system, # New multi-agent query function
    get_multi_agent_system # For checking system status
)
st.set_page_config(
    # Updated title to reflect broader scope
    page_title="ESI Academic & Z Assistant",
    page_icon="🎓", # Keep academic icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Using the Z assistant theme as base)
st.markdown("""
<style>
    .main {
        background-color: #0e1117; /* Dark background */
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #1e1e1e !important; /* Dark input fields */
        color: #ffffff !important; /* White text */
        border: 1px solid #80cbc4; /* Accent border */
    }
    .stFileUploader > div > div > button { /* Style file uploader button */
        background-color: #1e1e1e;
        color: #80cbc4;
        border: 1px solid #80cbc4;
    }
    .stFileUploader > div > div > button:hover {
         border-color: #ffffff;
         color: #ffffff;
    }


    .output-container {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px; /* Add space between outputs */
    }
    h1 {
        color: #4db6ac; /* Teal */
    }
    h2, h3 {
        color: #80cbc4; /* Light Teal */
    }
    .stButton>button {
        background-color: #26a69a; /* Teal button */
        color: white;
        border-radius: 5px;
        border: none; /* Remove default border */
    }
    .stButton>button:hover {
        background-color: #00897b; /* Darker Teal on hover */
    }
    .stRadio>label { /* Style radio button labels */
        color: #ffffff;
    }
    .z-notation { /* Specific style for Z notation if needed */
        font-family: "Cambria Math", "STIX", serif;
        background-color: #2a2a2a; /* Slightly different background for Z code? */
        padding: 5px;
        border-radius: 3px;
        display: block; /* Ensure it takes block space */
        white-space: pre-wrap; /* Preserve whitespace */
    }
    .schema-box { /* Style for Z schema boxes */
        border: 1px solid #80cbc4; /* Teal border */
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        background-color: #1a1a1a; /* Darker background inside box */
        white-space: pre-wrap; /* Preserve whitespace and line breaks */
        word-wrap: break-word; /* Wrap long lines */
        font-family: "Consolas", "Monaco", monospace; /* Monospace font for code/schema */
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #80cbc4;
        border-bottom: 2px solid transparent; /* Underline effect preparation */
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e1e1e !important; /* Keep background same */
        color: white !important;
        border-bottom: 2px solid #26a69a !important; /* Teal underline for active tab */
    }

    /* Styling chat history display */
    .chat-question {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 5px; /* Reduced space */
        border-left: 3px solid #26a69a; /* Teal border left for question */
        color: #e0e0e0; /* Lighter grey text */
    }
    .chat-response {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px; /* Space after response */
        border-left: 3px solid #4db6ac; /* Lighter teal border left for response */
        color: #ffffff; /* White text for response */
        overflow-x: auto; /* Allow horizontal scroll for wide content like code */
    }
    .chat-response strong { /* Make bold text in response stand out */
        color: #80cbc4;
    }
     .chat-response code { /* Style inline code */
        background-color: #333333;
        padding: 2px 4px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .chat-response pre code { /* Style code blocks */
        background-color: #1a1a1a !important; /* Darker background for code block */
        border-radius: 5px;
        padding: 10px;
        display: block;
        overflow-x: auto;
        border: 1px solid #333; /* Subtle border for code block */
    }

</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = [] # General history for the first tab
if 'z_analysis_output' not in st.session_state:
    st.session_state.z_analysis_output = None # Store last Z analysis result
if 'z_generation_output' not in st.session_state:
    st.session_state.z_generation_output = None # Store last Z generation result
if 'use_multi_agent' not in st.session_state:
    st.session_state.use_multi_agent = True # Default to using multi-agent system

# --- Helper Functions ---

def format_response(response_data):
    """Formats the response for display using Markdown."""
    if isinstance(response_data, dict):
        # Prioritize enhanced_response, then analysis, then raw, then string representation
        content = response_data.get('enhanced_response',
                    response_data.get('analysis',
                    response_data.get('raw_response', str(response_data))))
    else:
        content = str(response_data)

    # Use Streamlit's markdown rendering which handles code blocks etc.
    # The CSS above styles the markdown output (e.g., code blocks)
    return content # Return raw markdown string

def get_z_symbol_reference():
    """Return a reference table for Z notation symbols"""
    # Kept the Z symbol reference as it's specific and useful
    return """
# Syntaxe Détaillée du Langage Z

## 1. Introduction et Objectifs du Langage Z

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

Syntaxe :

* `/* Ceci est aussi un commentaire sur une ou plusieurs lignes */`

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
"""

# --- Main Application Logic ---
def main():
    # Sidebar Configuration
    with st.sidebar:
        st.title("🎓 ESI Assistant") # Updated title
        st.markdown("### Academic & Formal Methods Tool") # Updated subtitle
        st.markdown("---")

        st.markdown("### About")
        st.markdown("""
        This assistant uses RAG (Retrieval-Augmented Generation) to answer questions on various academic topics taught at ESI, including:
        - **Programming:** C, Assembly (8086), Algorithms, Data Structures
        - **Systems:** Operating Systems, Architecture
        - **Theory:** Formal Methods (special focus on **Z Notation**)
        - **And more...**

        Use the tabs for specific tasks:
        - **Academic Query:** General questions.
        - **Z Code Analysis:** Analyze Z specifications.
        - **Z Spec Generator:** Generate Z specs from requirements.
        """)

        st.markdown("---")

        # Language selection from old code
        st.markdown("### Preferences")
        language = st.radio(
            "Response Language:",
            ("English", "Français"), # Default English
            index=0,
            key="language_select"
        )
        # Map to language codes used by backend
        lang_code = "en" if language == "English" else "fr"
        
        # Add multi-agent system toggle
        st.markdown("### System Mode")
        use_multi_agent = st.toggle(
            "Use Multi-Agent System",
            value=st.session_state.use_multi_agent,
            help="Enable to use the multi-agent system with Groq retrieval and Gemini validation. Disable for standard RAG."
        )
        st.session_state.use_multi_agent = use_multi_agent
        
        # Show which system is active
        if use_multi_agent:
            st.success("Multi-Agent System Active")
            with st.expander("About Multi-Agent"):
                st.markdown("""
                **Multi-Agent Processing:**
                1. Coordinator analyzes your query
                2. Retriever gets relevant content
                3. Validator enhances & verifies
                4. Final response synthesis
                
                This approach combines the strengths of multiple AI systems to provide more accurate and comprehensive responses.
                """)
        else:
            st.info("Standard RAG System Active")

        st.markdown("---")

        # Z notation reference expander (still relevant)
        with st.expander("Z Notation Reference"):
            st.markdown(get_z_symbol_reference(), unsafe_allow_html=True)

        st.markdown("---")

        # Clear chat button
        if st.button("Clear Query History"):
            st.session_state.history = []
            # Optionally clear Z tool outputs too
            st.session_state.z_analysis_output = None
            st.session_state.z_generation_output = None
            st.success("Query history cleared.")
            st.rerun()  # Changed from st.experimental_rerun()

    # Main content - Tabbed interface
    tab1, tab2 = st.tabs(["Academic Query", "Z Code Analysis"])

    # --- Tab 1: Academic Query ---
    with tab1:
        st.header("Ask an Academic Question")

        # Chat history display area
        chat_container = st.container()
        with chat_container:
            if not st.session_state.history:
                 st.info("👋 Welcome! Ask a question about ESI subjects or upload a context file.")
                 st.markdown("""
                    **Example Questions:**
                    - Explain pointers in C with an example.
                    - Analyze the time complexity of Quicksort.
                    - How does virtual memory work?
                    - How do I define a schema in Z notation?
                    - Convert a string to uppercase in 8086 assembly.
                    """)
            else:
                for i, (q, a) in enumerate(st.session_state.history):
                    # Use custom CSS classes for styling
                    st.markdown(f"<div class='chat-question'><b>You:</b> {q}</div>", unsafe_allow_html=True)
                    # Use st.markdown for the response to render formatting
                    st.markdown(f"<div class='chat-response'><b>Assistant:</b><br>{format_response(a)}</div>", unsafe_allow_html=True)


        # Input area using st.form
        with st.form(key="query_form"):
            col1, col2 = st.columns([3, 1]) # Give more space to text area
            with col1:
                user_query_text = st.text_area(
                    "Your Question:",
                    height=100,
                    placeholder="Type your question here..."
                )
            with col2:
                 uploaded_file_query = st.file_uploader(
                    "Upload Context (.txt)",
                    type=["txt"],
                    key="query_file_uploader",
                    help="Optional: Upload a text file. Its content will be added as context to your question."
                )

            submit_button_query = st.form_submit_button("Send")

        # Process query submission
        if submit_button_query:
            final_query_content = user_query_text
            display_question = user_query_text
            file_context = ""

            if uploaded_file_query is not None:
                try:
                    file_context = uploaded_file_query.getvalue().decode("utf-8")
                    st.success(f"Using context from '{uploaded_file_query.name}'.")
                except Exception as e:
                    st.error(f"Error reading file '{uploaded_file_query.name}': {e}")
                    file_context = "" # Reset on error

            # Combine query and context for the backend
            if file_context:
                if user_query_text:
                    final_query_content = f"Context from file '{uploaded_file_query.name}':\n```\n{file_context}\n```\n\nUser Question: {user_query_text}"
                    display_question = f"{user_query_text} (with context from {uploaded_file_query.name})"
                else:
                    # If only file is uploaded, ask the backend to process the file content
                    final_query_content = f"Please analyze, summarize, or answer questions based on the following text from the uploaded file '{uploaded_file_query.name}':\n```\n{file_context}\n```"
                    display_question = f"Query based on uploaded file: {uploaded_file_query.name}"

            if not final_query_content:
                st.warning("Please enter a question or upload a file.")
            else:
                # Call the appropriate query function based on user selection
                with st.spinner(f"{'Multi-agent processing' if st.session_state.use_multi_agent else 'Thinking'} in {language}..."):
                    try:
                        if st.session_state.use_multi_agent:
                            # Use the multi-agent system directly
                            response = query_multi_agent_system(
                                query=final_query_content,
                                language=lang_code
                            )
                        else:
                            # Use standard academic RAG
                            response = query_academic_rag(
                                query=final_query_content,
                                subject=None, # Let backend detect
                                language=lang_code
                            )

                        # Add to history
                        st.session_state.history.append((display_question, response)) # Store the dict

                        # Rerun to display the new message at the top of the chat container
                        st.rerun()  # Changed from st.experimental_rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    # --- Tab 2: Z Code Analysis ---
    with tab2:
        st.header("Z Specification Analyzer")
        st.markdown("Enter Z specification code below or upload a `.txt` or `.alg` file for analysis.")

        with st.form(key="z_code_form"):
            z_code_from_text_area = st.text_area(
                "Enter Z Specification:",
                height=350, # Increased height
                placeholder="Example:\n Ecris Code Z pour faire la somme de deux entiers.\n\n[...]\n\n"
                

            )
            uploaded_file_z_code = st.file_uploader(
                "Upload Z File (.txt, .alg)",
                type=["txt", "alg"],
                key="z_code_file_uploader"
            )
            submit_code_button = st.form_submit_button("Analyze Specification")

        # Processing analysis request
        if submit_code_button:
            z_code_to_analyze = ""
            source_info = ""

            if uploaded_file_z_code is not None:
                try:
                    z_code_to_analyze = uploaded_file_z_code.getvalue().decode("utf-8")
                    source_info = f"Analysis based on uploaded file: `{uploaded_file_z_code.name}`"
                    st.success(f"Read file '{uploaded_file_z_code.name}'.")
                except Exception as e:
                    st.error(f"Error reading file '{uploaded_file_z_code.name}': {e}")
            elif z_code_from_text_area:
                z_code_to_analyze = z_code_from_text_area
                source_info = "Analysis based on text area input."
            else:
                st.warning("Please provide Z specification via text area or file upload.")

            if z_code_to_analyze:
                st.info(source_info)
                with st.spinner("Analyzing Z specification..."):
                    try:
                        analysis_result = analyze_z_code(z_code_to_analyze)
                        # Store the result in session state for display
                        st.session_state.z_analysis_output = analysis_result
                        st.rerun()  # Changed from st.experimental_rerun() - Rerun to display results cleanly below the form

                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        st.session_state.z_analysis_output = None # Clear output on error

        # Display analysis results if available in session state
        if st.session_state.z_analysis_output:
            st.markdown("---")
            st.subheader("Analysis Results")
            analysis_content = format_response(st.session_state.z_analysis_output)
            st.markdown(analysis_content, unsafe_allow_html=True) # Use markdown rendering

            suggestions = st.session_state.z_analysis_output.get('suggestions', [])
            if suggestions and suggestions != ["No specific code suggestions extracted."]:
                st.subheader("Suggestions")
                for i, suggestion in enumerate(suggestions):
                    with st.expander(f"Suggestion {i+1}"):
                        # Display suggestion as code block
                         st.code(suggestion, language='text') # Use 'text' or 'z' if supported


   
# --- RAG System Loading ---
if 'rag_system_loaded' not in st.session_state:
    st.session_state.rag_system_loaded = False
if 'multi_agent_loaded' not in st.session_state:
    st.session_state.multi_agent_loaded = False

# Load only once per session
if not st.session_state.rag_system_loaded:
    with st.spinner("Loading RAG knowledge base... Please wait."):
        try:
            load_rag_system() # Loads the standard RAG system
            st.session_state.rag_system_loaded = True
            
            # Also initialize multi-agent system in background
            try:
                multi_agent = get_multi_agent_system()
                if multi_agent.is_initialized:
                    st.session_state.multi_agent_loaded = True
                st.success("Multi-agent system initialized")
            except Exception as e:
                st.warning(f"Multi-agent system couldn't be initialized: {str(e)}. Standard RAG will be used instead.")
            
            # Display success message only on the *first* successful load
            if not hasattr(st.session_state, 'initial_load_success_shown'):
                 st.success("Systems loaded successfully!")
                 st.session_state.initial_load_success_shown = True
        except Exception as e:
            st.error(f"FATAL: Failed to load RAG system: {str(e)}")
            st.stop() # Stop the app if RAG system fails to load


# --- Run the main app ---
if st.session_state.rag_system_loaded:
    main()
else:
    # This part should ideally not be reached if st.stop() works correctly on error
    st.error("RAG system is not loaded. Cannot start the application.")

# --- END OF FILE app.py --- (Merged Version)