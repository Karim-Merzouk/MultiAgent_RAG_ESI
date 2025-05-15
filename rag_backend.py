import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import google.generativeai as genai

import os
import traceback
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# API Keys - IMPORTANT: Use environment variables or Streamlit secrets in production for security
# For Groq, set the environment variable directly. For Google, we'll use a variable.
load_dotenv()  # Load environment variables from .env file

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Global variables for loaded components
_loaded_retriever = None
_loaded_llm = None

# --- Z Language Specific Prompt (from original Z-focused backend) ---
def create_z_language_prompt_template():
        
    return ChatPromptTemplate.from_template('''[Context] {context}
        [Question] {input}

        **Instructions for Z Language Analysis**:

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

        Les instructions décrivent les opérations à effectuer.''')

# --- Multi-Agent System Architecture ---
# Agent Roles and System Prompts
AGENT_SYSTEM_PROMPTS = {
    "coordinator": """You are the Coordinator Agent responsible for:
1. Analyzing user queries to determine appropriate retrieval strategy
2. Dispatching tasks to specialized agents
3. Synthesizing their outputs into cohesive responses
4. Managing the overall conversation flow
5. Ensuring high quality, accurate responses through agent validation
""",
    "groq_retriever": """You are the Retrieval Agent powered by Groq's model.
Your responsibilities include:
1. Retrieving relevant information from the vector database
2. Analyzing and understanding the retrieved content
3. Generating initial responses based on the retrieved knowledge
4. Identifying areas where additional information may be needed
5. Providing clear explanations focused on academic quality and accuracy
""",
    "gemini_validator": """You are the Validation Agent powered by Google's Gemini model.
Your responsibilities include:
1. Carefully reviewing responses from the Retrieval Agent
2. Validating factual accuracy of the information
3. Enhancing responses with additional context, examples, or clarifications
4. Ensuring pedagogical quality appropriate for an ESI student audience
5. Restructuring content for clarity, completeness, and correctness
6. Flagging potential inaccuracies or misconceptions for correction
"""
}

# --- Natural Language Querying and Code Generation ---
def query_esi_rag(query, system_mode="auto", specified_subject_hint=None, language="en"):
    global _loaded_retriever, _loaded_llm
    if not _loaded_retriever or not _loaded_llm:
        load_rag_system() # Ensure system is loaded

    try:
        intent = "concept_explanation"
        subject = None

        if system_mode == "auto":
            intent = detect_user_intent(query)
            subject = detect_subject_area(query, specified_subject_hint)
        elif system_mode in SUBJECT_PROMPTS: # system_mode can be a subject name like "Z Notation"
            subject = system_mode
            intent = detect_user_intent(query) # Still detect intent for Gemini context
        elif system_mode in ESI_PROMPT_TEMPLATES: # system_mode can be an intent name
            intent = system_mode
            subject = detect_subject_area(query, specified_subject_hint)
        
        # Select prompt with better error handling
        selected_prompt = None
        if subject and subject in SUBJECT_PROMPTS:
            selected_prompt = SUBJECT_PROMPTS[subject]
        elif intent and intent in ESI_PROMPT_TEMPLATES:
            selected_prompt = ESI_PROMPT_TEMPLATES[intent]
        else:
            selected_prompt = ESI_PROMPT_TEMPLATES["concept_explanation"] # Fallback
            
        # Ensure we have a valid prompt template
        if selected_prompt is None:
            st.error("Failed to select a prompt template. Using default.")
            selected_prompt = ChatPromptTemplate.from_template('''
            [Context] {context}
            [Question] {input}
            
            Please answer this question based on the provided context and your knowledge.
            ''')
        
        # Create chain and execute
        document_chain = create_stuff_documents_chain(_loaded_llm, selected_prompt)
        current_chain = create_retrieval_chain(_loaded_retriever, document_chain)
        
        # Handle language for the query to Groq if needed (Groq usually infers)
        # For Gemini, language handling is done in get_gemini_response or by prepending to query
        query_for_llm = query
        if language.lower() == "fr":
             # Prepend to query for Gemini, Groq might not need it explicitly
            query_for_llm = f"[Répondre en français s'il vous plaît] {query}"


        result = current_chain.invoke({'input': query_for_llm})
        initial_answer = result['answer']
        
        enhanced_answer = get_gemini_response(
            initial_answer,
            query=query, # Original query for Gemini context
            intent=intent,
            subject=subject
        )
        
        return {
            'raw_response': initial_answer,
            'enhanced_response': enhanced_answer,
            'relevant_documents': result.get('context', []),
            'detected_intent': intent,
            'detected_subject': subject
        }
    except Exception as e:
        raise Exception(f"Error querying ESI RAG system: {str(e)}\n{traceback.format_exc()}")

# --- Z Language Specific Functions (adapted) ---
def query_z_language_rag(query):
    """Query the RAG system specifically for Z language information."""
    # This function now uses query_esi_rag with "Z Notation" as the system_mode/subject
    # to ensure the Z-specific prompt from the dictionary is used.
    return query_esi_rag(query, system_mode="Z Notation", specified_subject_hint="Z Notation")

def extract_code_suggestions(analysis_text, language="text"): # language added for ```language
    try:
        suggestions = []
        # Regex to find code blocks, including optional language specifier
        # Matches ``` optionally followed by a language name, then content, then ```
        code_block_pattern = re.compile(r"```(?:[a-zA-Z0-9_.-]*\n)?(.*?)```", re.DOTALL)
        
        # Find all code blocks in the analysis text
        matches = code_block_pattern.finditer(analysis_text)
        for match in matches:
            # The actual code is in group 1
            code_content = match.group(1).strip()
            if code_content: # Ensure there's content in the block
                suggestions.append(code_content)
        
        # If no formal code blocks found, try a simpler line-based extraction (less reliable)
        if not suggestions:
            lines = analysis_text.split('\n')
            in_code_block = False
            current_suggestion = []
            for line in lines:
                if line.strip().startswith("```") and not in_code_block:
                    in_code_block = True
                    # Skip the opening ``` line itself from being added to suggestion
                    if len(line.strip()) > 3: # handles ```z or ```c
                        pass # language specifier is part of the ``` line
                    continue 
                elif line.strip() == "```" and in_code_block:
                    in_code_block = False
                    if current_suggestion:
                        suggestions.append("\n".join(current_suggestion))
                        current_suggestion = []
                    continue
                if in_code_block:
                    current_suggestion.append(line)
            if current_suggestion: # capture any trailing block
                 suggestions.append("\n".join(current_suggestion))

        return suggestions if suggestions else ["No specific code suggestions extracted."]
    except Exception as e:
        return [f"Error extracting suggestions: {str(e)}"]


def analyze_z_code(z_code):
    """Analyze Z language code for correctness and optimization"""
    try:
        context_result = query_z_language_rag("Z language best practices, common syntax patterns, and schema analysis techniques.")
        
        # Use Gemini to specifically analyze the provided code, with context
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert Z language instructor and formal methods specialist.
        
        Analyze the following Z specification:
        ```z
        {z_code}
        ```
        
        Consider these Z language best practices and common patterns for context:
        {context_result['enhanced_response']}
        
        Provide a detailed analysis covering:
        1. Syntax correctness and adherence to Z notation standards.
        2. Semantic clarity and potential ambiguities.
        3. Type consistency and correctness of declarations.
        4. Soundness of predicates and invariants.
        5. Completeness and appropriateness of operations defined.
        6. Suggestions for improvement, simplification, or alternative formulations, with examples.
        7. Potential errors or common pitfalls evident in the specification.

        Format your response clearly, using markdown for structure. Highlight specific parts of the Z code when discussing them.
        """
        response = model.generate_content(prompt)
        analysis = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)
        
        return {
            'analysis': analysis,
            'suggestions': extract_code_suggestions(analysis, "z")
        }
    except Exception as e:
        raise Exception(f"Error analyzing Z code: {str(e)}")

def generate_optimized_z_code(specification_text):
    """Generate optimized Z language code based on a natural language specification"""
    try:
        context_result = query_z_language_rag("Z language formal specification patterns, schema design principles, and optimization techniques for clarity and rigor.")
        
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
        prompt = f"""
        You are an expert Z language formal specification writer.
        
        Based on this context about Z language:
        {context_result['enhanced_response']}
        
        Generate a formal Z specification for the following requirements:
        "{specification_text}"
        
        Follow these guidelines:
        1. Use proper Z syntax and conventions.
        2. Include helpful comments explaining key sections or complex logic.
        3. Follow best practices for Z programming (e.g., error handling, resource management if applicable).
        4. Make the code efficient and readable.
        5. Use appropriate data structures and algorithms from the standard library if available and suitable.
        
        Return the Z specification code formatted within ```z ... ``` blocks.
        Also, provide a brief explanation of your implementation choices, data structures used, and any important considerations.
        """
        response = model.generate_content(prompt)
        full_response_text = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)

        # Extract Z code from the response
        z_code_match = re.search(r"```z\n(.*?)\n```", full_response_text, re.DOTALL)
        generated_code = z_code_match.group(1).strip() if z_code_match else "No Z specification code block found in the response."
        
        # The rest of the text can be considered an explanation or analysis
        explanation = re.sub(r"```z\n(.*?)\n```", "", full_response_text, flags=re.DOTALL).strip()
        if not explanation and generated_code != "No Z specification code block found in the response.":
            explanation = "Z specification generated as requested."


        return {
            'generated_code': generated_code,
            'analysis': explanation if explanation else "No additional analysis provided.",
            'optimized_suggestions': extract_code_suggestions(explanation, "z") # Suggestions from the explanation part
        }
    except Exception as e:
        raise Exception(f"Error generating Z code: {str(e)}")

# --- General Code Analysis and Generation (for other languages) ---
def analyze_code(code, language):
    """Analyze code for a specific programming language"""
    try:
        context_result = query_esi_rag(
            f"{language} programming best practices, common coding patterns, and debugging techniques.",
            specified_subject_hint=f"{language} Programming"
        )
        
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert ESI instructor specializing in {language} programming.
        
        Analyze this {language} code:
        ```{language.lower()}
        {code}
        ```
        
        Context from ESI knowledge base about {language} best practices:
        {context_result['enhanced_response']}
        
        Provide:
        1. A detailed code analysis including structure, style, and organization.
        2. Identification of any bugs, logical errors, or suboptimal implementations.
        3. Suggestions for improvement with specific, corrected code examples.
        4. Explanation of key concepts or algorithms demonstrated in the code.
        5. Adherence to {language} best practices and idiomatic expressions.
        
        Format your response with clear sections and provide corrected code examples within ```{language.lower()} ... ``` blocks.
        """
        response = model.generate_content(prompt)
        analysis = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)
        
        return {
            'analysis': analysis,
            'suggestions': extract_code_suggestions(analysis, language.lower())
        }
    except Exception as e:
        raise Exception(f"Error analyzing {language} code: {str(e)}")

def generate_optimized_code(requirements, language):
    """Generate optimized code based on natural language requirements"""
    try:
        context_result = query_esi_rag(
            f"{language} programming best practices, standard library usage, and optimization techniques for common tasks.",
            specified_subject_hint=f"{language} Programming"
        )
        
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert {language} programmer and ESI instructor.
        
        Based on this context about {language} programming:
        {context_result['enhanced_response']}
        
        Generate clean, well-structured, and idiomatic {language} code for the following requirements:
        "{requirements}"
        
        Follow these guidelines:
        1. Use proper {language} syntax and conventions.
        2. Include helpful comments explaining key sections or complex logic.
        3. Follow best practices for {language} programming (e.g., error handling, resource management if applicable).
        4. Make the code efficient and readable.
        5. Use appropriate data structures and algorithms from the standard library if available and suitable.
        
        Return the generated {language} code within ```{language.lower()} ... ``` blocks.
        Also, provide a brief explanation of your implementation choices, data structures used, and any important considerations.
        """
        response = model.generate_content(prompt)
        full_response_text = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)

        # Extract code from the response
        code_match = re.search(rf"```{language.lower()}\n(.*?)\n```", full_response_text, re.DOTALL)
        generated_code = code_match.group(1).strip() if code_match else f"No {language} code block found."
        
        explanation = re.sub(rf"```{language.lower()}\n(.*?)\n```", "", full_response_text, flags=re.DOTALL).strip()
        if not explanation and generated_code != f"No {language} code block found.":
            explanation = f"{language} code generated as per requirements."

        return {
            'generated_code': generated_code,
            'full_response': full_response_text, # For debugging or fuller context
            'analysis': explanation if explanation else "No additional analysis provided.",
            'optimization_suggestions': [] # Suggestions would typically come from analyzing this generated code, or be part of the explanation
        }
    except Exception as e:
        raise Exception(f"Error generating {language} code: {str(e)}")

# --- Academic Query Wrapper ---
def query_academic_rag(query, subject, language="en"):
    """Query the RAG system specifically for academic subjects with language selection"""
    # Use the multi-agent system for complex queries or when explicitly requested
    use_multi_agent = False
    
    # Check if query is complex enough to warrant multi-agent processing
    if len(query) > 50 or "explain in detail" in query.lower() or "analyze" in query.lower():
        use_multi_agent = True
    
    if use_multi_agent:
        try:
            return query_multi_agent_system(query, language)
        except Exception as e:
            st.warning(f"Multi-agent system failed, falling back to standard RAG: {str(e)}")
            # Fall back to standard RAG if multi-agent fails
    
    # Use standard RAG otherwise
    return query_esi_rag(query, specified_subject_hint=subject, language=language)  # Ensure proper newline

# --- Core RAG System Loading and Helper Functions ---
def load_embeddings():
    return SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True
        }
    )

def load_vectorstore(embeddings):
    save_directory = "./faiss_index"
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"FAISS index not found at {save_directory}. Please run the indexing script first.")
    return FAISS.load_local(
        save_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )

def get_groq_llm():
    return ChatGroq(
        model="deepseek-r1-distill-llama-70b", # A capable model available on Groq, was deepseek-r1-distill-llama-70b
        temperature=0.1,
        max_tokens=8192
    )

def get_gemini_response(text, query=None, intent=None, subject=None):
    try:
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05") # Using a common Gemini model
        
        prompt = f"""
        You are an expert ESI (École Supérieure d'Informatique) instructor specializing in computer science education.
        
        Base information / context:
        {text}
        
        User query: {query if query else "No specific query provided, general context enhancement."}
        
        Detected Intent: {intent if intent else "General explanation"}
        Detected Subject area: {subject if subject else "General computer science"}
        
        Provide a comprehensive and pedagogically sound educational response based on the user query and context:
        1. Directly address the user's question with academic rigor.
        2. Elaborate on the provided base information, adding explanations, examples, or clarifications as needed.
        3. If the base information seems to be an answer from another AI, refine it, ensure its accuracy, and present it in your own expert instructor voice.
        4. Include clear explanations of theoretical concepts.
        5. Provide practical examples and code snippets (e.g., C,8086 architechture Assembly, Z Notation, Python, Java) when relevant to the subject and query.
        6. Use proper technical terminology and notation.
        7. Structure information clearly with headings, bullet points, and markdown formatting (including code blocks like ```language ... ```).
        
        Ensure your response is tailored to an ESI student audience.
        """
        
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        elif response.parts:
            return "".join(part.text for part in response.parts)
        else: # Fallback for unexpected response structure
            st.warning("Gemini response structure not as expected. Raw response parts might be missing.")
            return str(response.candidates[0].content.parts[0]) if response.candidates and response.candidates[0].content.parts else text

    except Exception as e:
        st.error(f"Gemini processing failed: {str(e)}")
        return text # Return original text if Gemini fails

@st.cache_resource
def load_rag_system():
    global _loaded_retriever, _loaded_llm
    if _loaded_retriever and _loaded_llm:
        return True
    try:
        embeddings = load_embeddings()
        db = load_vectorstore(embeddings)
        _loaded_retriever = db.as_retriever(search_kwargs={"k": 7})
        _loaded_llm = get_groq_llm()
        return True
    except Exception as e:
        raise Exception(f"Failed to load RAG system: {str(e)}\n{traceback.format_exc()}")

# --- Intent and Subject Detection ---
def detect_user_intent(query):
    query_lower = query.lower()
    intent_patterns = {
        "concept_explanation": [r'explain', r'what is', r'define', r'describe', r'concept of', r'mean'],
        "algorithm_analysis": [r'analyze', r'complexity of', r'big o', r'performance of', r'algorithm for', r'efficiency'],
        "code_debugging": [r'debug', r'error', r'fix', r'problem with', r'incorrect', r'doesn\'t work', r'not working'],
        "exercise_solution": [r'solve', r'solution', r'exercise', r'problem', r'assignment', r'homework'],
        "exam_preparation": [r'prepare', r'exam', r'study', r'review', r'key points', r'important concepts'],
        "project_guidance": [r'project', r'develop', r'implement', r'create', r'build', r'design', r'architecture'],
        "formal_methods": [r'formal', r'proof', r'verify', r'specification', r'correctness', r'z notation', r'schema'],
        "lecture_summary": [r'summarize', r'summary', r'overview', r'lecture', r'course', r'lesson'],
        "exercise_walkthrough": [r'walkthrough', r'step by step', r'solve this', r'work through'],
        "concept_review": [r'review', r'revise', r'revisit', r'understand'],
        "code_explanation": [r'explain code', r'what does this code', r'how does this code']
    }
    intent_scores = {intent: 0 for intent in intent_patterns}
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                intent_scores[intent] += 1
    max_score = max(intent_scores.values())
    if max_score > 0:
        for intent, score in intent_scores.items():
            if score == max_score: return intent
    return "concept_explanation"

def detect_subject_area(query, specified_subject=None):
    if specified_subject and specified_subject in SUBJECT_PROMPTS:
        return specified_subject # Prioritize explicitly passed subject if valid

    query_lower = query.lower()
    subject_scores = {subject: 0 for subject in ESI_TOPICS}
    for subject, keywords in ESI_TOPICS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                subject_scores[subject] += 1
    
    # Check if any of the SUBJECT_PROMPTS keys are in the query
    for subj_key in SUBJECT_PROMPTS.keys():
        if subj_key.lower() in query_lower:
            subject_scores[subj_key] = subject_scores.get(subj_key, 0) + 5 # Boost score for direct match

    max_score = max(subject_scores.values()) if subject_scores else 0
    if max_score > 0:
        for subject, score in subject_scores.items():
            if score == max_score: return subject
    return None

# --- Core Querying Logic ---
def query_esi_rag(query, system_mode="auto", specified_subject_hint=None, language="en"):
    global _loaded_retriever, _loaded_llm
    if not _loaded_retriever or not _loaded_llm:
        load_rag_system() # Ensure system is loaded

    try:
        intent = "concept_explanation"
        subject = None

        if system_mode == "auto":
            intent = detect_user_intent(query)
            subject = detect_subject_area(query, specified_subject_hint)
        elif system_mode in SUBJECT_PROMPTS: # system_mode can be a subject name like "Z Notation"
            subject = system_mode
            intent = detect_user_intent(query) # Still detect intent for Gemini context
        elif system_mode in ESI_PROMPT_TEMPLATES: # system_mode can be an intent name
            intent = system_mode
            subject = detect_subject_area(query, specified_subject_hint)
        
        # Select prompt with better error handling
        selected_prompt = None
        if subject and subject in SUBJECT_PROMPTS:
            selected_prompt = SUBJECT_PROMPTS[subject]
        elif intent and intent in ESI_PROMPT_TEMPLATES:
            selected_prompt = ESI_PROMPT_TEMPLATES[intent]
        else:
            selected_prompt = ESI_PROMPT_TEMPLATES["concept_explanation"] # Fallback
            
        # Ensure we have a valid prompt template
        if selected_prompt is None:
            st.error("Failed to select a prompt template. Using default.")
            selected_prompt = ChatPromptTemplate.from_template('''
            [Context] {context}
            [Question] {input}
            
            Please answer this question based on the provided context and your knowledge.
            ''')
        
        # Create chain and execute
        document_chain = create_stuff_documents_chain(_loaded_llm, selected_prompt)
        current_chain = create_retrieval_chain(_loaded_retriever, document_chain)
        
        # Handle language for the query to Groq if needed (Groq usually infers)
        # For Gemini, language handling is done in get_gemini_response or by prepending to query
        query_for_llm = query
        if language.lower() == "fr":
             # Prepend to query for Gemini, Groq might not need it explicitly
            query_for_llm = f"[Répondre en français s'il vous plaît] {query}"


        result = current_chain.invoke({'input': query_for_llm})
        initial_answer = result['answer']
        
        enhanced_answer = get_gemini_response(
            initial_answer,
            query=query, # Original query for Gemini context
            intent=intent,
            subject=subject
        )
        
        return {
            'raw_response': initial_answer,
            'enhanced_response': enhanced_answer,
            'relevant_documents': result.get('context', []),
            'detected_intent': intent,
            'detected_subject': subject
        }
    except Exception as e:
        raise Exception(f"Error querying ESI RAG system: {str(e)}\n{traceback.format_exc()}")

# --- Z Language Specific Functions (adapted) ---
def query_z_language_rag(query):
    """Query the RAG system specifically for Z language information."""
    # This function now uses query_esi_rag with "Z Notation" as the system_mode/subject
    # to ensure the Z-specific prompt from the dictionary is used.
    return query_esi_rag(query, system_mode="Z Notation", specified_subject_hint="Z Notation")

def extract_code_suggestions(analysis_text, language="text"): # language added for ```language
    try:
        suggestions = []
        # Regex to find code blocks, including optional language specifier
        # Matches ``` optionally followed by a language name, then content, then ```
        code_block_pattern = re.compile(r"```(?:[a-zA-Z0-9_.-]*\n)?(.*?)```", re.DOTALL)
        
        # Find all code blocks in the analysis text
        matches = code_block_pattern.finditer(analysis_text)
        for match in matches:
            # The actual code is in group 1
            code_content = match.group(1).strip()
            if code_content: # Ensure there's content in the block
                suggestions.append(code_content)
        
        # If no formal code blocks found, try a simpler line-based extraction (less reliable)
        if not suggestions:
            lines = analysis_text.split('\n')
            in_code_block = False
            current_suggestion = []
            for line in lines:
                if line.strip().startswith("```") and not in_code_block:
                    in_code_block = True
                    # Skip the opening ``` line itself from being added to suggestion
                    if len(line.strip()) > 3: # handles ```z or ```c
                        pass # language specifier is part of the ``` line
                    continue 
                elif line.strip() == "```" and in_code_block:
                    in_code_block = False
                    if current_suggestion:
                        suggestions.append("\n".join(current_suggestion))
                        current_suggestion = []
                    continue
                if in_code_block:
                    current_suggestion.append(line)
            if current_suggestion: # capture any trailing block
                 suggestions.append("\n".join(current_suggestion))

        return suggestions if suggestions else ["No specific code suggestions extracted."]
    except Exception as e:
        return [f"Error extracting suggestions: {str(e)}"]


def analyze_z_code(z_code):
    """Analyze Z language code for correctness and optimization"""
    try:
        context_result = query_z_language_rag("Z language best practices, common syntax patterns, and schema analysis techniques.")
        
        # Use Gemini to specifically analyze the provided code, with context
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert Z language instructor and formal methods specialist.
        
        Analyze the following Z specification:
        ```z
        {z_code}
        ```
        
        Consider these Z language best practices and common patterns for context:
        {context_result['enhanced_response']}
        
        Provide a detailed analysis covering:
        1. Syntax correctness and adherence to Z notation standards.
        2. Semantic clarity and potential ambiguities.
        3. Type consistency and correctness of declarations.
        4. Soundness of predicates and invariants.
        5. Completeness and appropriateness of operations defined.
        6. Suggestions for improvement, simplification, or alternative formulations, with examples.
        7. Potential errors or common pitfalls evident in the specification.

        Format your response clearly, using markdown for structure. Highlight specific parts of the Z code when discussing them.
        """
        response = model.generate_content(prompt)
        analysis = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)
        
        return {
            'analysis': analysis,
            'suggestions': extract_code_suggestions(analysis, "z")
        }
    except Exception as e:
        raise Exception(f"Error analyzing Z code: {str(e)}")

def generate_optimized_z_code(specification_text):
    """Generate optimized Z language code based on a natural language specification"""
    try:
        context_result = query_z_language_rag("Z language formal specification patterns, schema design principles, and optimization techniques for clarity and rigor.")
        
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
        prompt = f"""
        You are an expert Z language formal specification writer.
        
        Based on this context about Z language:
        {context_result['enhanced_response']}
        
        Generate a formal Z specification for the following requirements:
        "{specification_text}"
        
        Follow these guidelines:
        1. Use proper Z syntax and conventions.
        2. Include helpful comments explaining key sections or complex logic.
        3. Follow best practices for Z programming (e.g., error handling, resource management if applicable).
        4. Make the code efficient and readable.
        5. Use appropriate data structures and algorithms from the standard library if available and suitable.
        
        Return the Z specification code formatted within ```z ... ``` blocks.
        Also, provide a brief explanation of your implementation choices, data structures used, and any important considerations.
        """
        response = model.generate_content(prompt)
        full_response_text = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)

        # Extract Z code from the response
        z_code_match = re.search(r"```z\n(.*?)\n```", full_response_text, re.DOTALL)
        generated_code = z_code_match.group(1).strip() if z_code_match else "No Z specification code block found in the response."
        
        # The rest of the text can be considered an explanation or analysis
        explanation = re.sub(r"```z\n(.*?)\n```", "", full_response_text, flags=re.DOTALL).strip()
        if not explanation and generated_code != "No Z specification code block found in the response.":
            explanation = "Z specification generated as requested."


        return {
            'generated_code': generated_code,
            'analysis': explanation if explanation else "No additional analysis provided.",
            'optimized_suggestions': extract_code_suggestions(explanation, "z") # Suggestions from the explanation part
        }
    except Exception as e:
        raise Exception(f"Error generating Z code: {str(e)}")

# --- General Code Analysis and Generation (for other languages) ---
def analyze_code(code, language):
    """Analyze code for a specific programming language"""
    try:
        context_result = query_esi_rag(
            f"{language} programming best practices, common coding patterns, and debugging techniques.",
            specified_subject_hint=f"{language} Programming"
        )
        
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert ESI instructor specializing in {language} programming.
        
        Analyze this {language} code:
        ```{language.lower()}
        {code}
        ```
        
        Context from ESI knowledge base about {language} best practices:
        {context_result['enhanced_response']}
        
        Provide:
        1. A detailed code analysis including structure, style, and organization.
        2. Identification of any bugs, logical errors, or suboptimal implementations.
        3. Suggestions for improvement with specific, corrected code examples.
        4. Explanation of key concepts or algorithms demonstrated in the code.
        5. Adherence to {language} best practices and idiomatic expressions.
        
        Format your response with clear sections and provide corrected code examples within ```{language.lower()} ... ``` blocks.
        """
        response = model.generate_content(prompt)
        analysis = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)
        
        return {
            'analysis': analysis,
            'suggestions': extract_code_suggestions(analysis, language.lower())
        }
    except Exception as e:
        raise Exception(f"Error analyzing {language} code: {str(e)}")

def generate_optimized_code(requirements, language):
    """Generate optimized code based on natural language requirements"""
    try:
        context_result = query_esi_rag(
            f"{language} programming best practices, standard library usage, and optimization techniques for common tasks.",
            specified_subject_hint=f"{language} Programming"
        )
        
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert {language} programmer and ESI instructor.
        
        Based on this context about {language} programming:
        {context_result['enhanced_response']}
        
        Generate clean, well-structured, and idiomatic {language} code for the following requirements:
        "{requirements}"
        
        Follow these guidelines:
        1. Use proper {language} syntax and conventions.
        2. Include helpful comments explaining key sections or complex logic.
        3. Follow best practices for {language} programming (e.g., error handling, resource management if applicable).
        4. Make the code efficient and readable.
        5. Use appropriate data structures and algorithms from the standard library if available and suitable.
        
        Return the generated {language} code within ```{language.lower()} ... ``` blocks.
        Also, provide a brief explanation of your implementation choices, data structures used, and any important considerations.
        """
        response = model.generate_content(prompt)
        full_response_text = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)

        # Extract code from the response
        code_match = re.search(rf"```{language.lower()}\n(.*?)\n```", full_response_text, re.DOTALL)
        generated_code = code_match.group(1).strip() if code_match else f"No {language} code block found."
        
        explanation = re.sub(rf"```{language.lower()}\n(.*?)\n```", "", full_response_text, flags=re.DOTALL).strip()
        if not explanation and generated_code != f"No {language} code block found.":
            explanation = f"{language} code generated as per requirements."

        return {
            'generated_code': generated_code,
            'full_response': full_response_text, # For debugging or fuller context
            'analysis': explanation if explanation else "No additional analysis provided.",
            'optimization_suggestions': [] # Suggestions would typically come from analyzing this generated code, or be part of the explanation
        }
    except Exception as e:
        raise Exception(f"Error generating {language} code: {str(e)}")

# --- Academic Query Wrapper ---
def query_academic_rag(query, subject, language="en"):
    """Query the RAG system specifically for academic subjects with language selection"""
    # Use the multi-agent system for complex queries or when explicitly requested
    use_multi_agent = False
    
    # Check if query is complex enough to warrant multi-agent processing
    if len(query) > 50 or "explain in detail" in query.lower() or "analyze" in query.lower():
        use_multi_agent = True
    
    if use_multi_agent:
        try:
            return query_multi_agent_system(query, language)
        except Exception as e:
            st.warning(f"Multi-agent system failed, falling back to standard RAG: {str(e)}")
            # Fall back to standard RAG if multi-agent fails
    
    # Use standard RAG otherwise
    return query_esi_rag(query, specified_subject_hint=subject, language=language)  # Ensure proper newline

# --- Core RAG System Loading and Helper Functions ---
def load_embeddings():
    return SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True
        }
    )

def load_vectorstore(embeddings):
    save_directory = "./faiss_index"
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"FAISS index not found at {save_directory}. Please run the indexing script first.")
    return FAISS.load_local(
        save_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )

def get_groq_llm():
    return ChatGroq(
        model="deepseek-r1-distill-llama-70b", # A capable model available on Groq, was deepseek-r1-distill-llama-70b
        temperature=0.1,
        max_tokens=8192
    )

def get_gemini_response(text, query=None, intent=None, subject=None):
    try:
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05") # Using a common Gemini model
        
        prompt = f"""
        You are an expert ESI (École Supérieure d'Informatique) instructor specializing in computer science education.
        
        Base information / context:
        {text}
        
        User query: {query if query else "No specific query provided, general context enhancement."}
        
        Detected Intent: {intent if intent else "General explanation"}
        Detected Subject area: {subject if subject else "General computer science"}
        
        Provide a comprehensive and pedagogically sound educational response based on the user query and context:
        1. Directly address the user's question with academic rigor.
        2. Elaborate on the provided base information, adding explanations, examples, or clarifications as needed.
        3. If the base information seems to be an answer from another AI, refine it, ensure its accuracy, and present it in your own expert instructor voice.
        4. Include clear explanations of theoretical concepts.
        5. Provide practical examples and code snippets (e.g., C, 8086 Assembly code, Z Notation, Python, Java) when relevant to the subject and query.
        6. Use proper technical terminology and notation.
        7. Structure information clearly with headings, bullet points, and markdown formatting (including code blocks like ```language ... ```).
        
        Ensure your response is tailored to an ESI student audience.
        """
        
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        elif response.parts:
            return "".join(part.text for part in response.parts)
        else: # Fallback for unexpected response structure
            st.warning("Gemini response structure not as expected. Raw response parts might be missing.")
            return str(response.candidates[0].content.parts[0]) if response.candidates and response.candidates[0].content.parts else text

    except Exception as e:
        st.error(f"Gemini processing failed: {str(e)}")
        return text # Return original text if Gemini fails

@st.cache_resource
def load_rag_system():
    global _loaded_retriever, _loaded_llm
    if _loaded_retriever and _loaded_llm:
        return True
    try:
        embeddings = load_embeddings()
        db = load_vectorstore(embeddings)
        _loaded_retriever = db.as_retriever(search_kwargs={"k": 7})
        _loaded_llm = get_groq_llm()
        return True
    except Exception as e:
        raise Exception(f"Failed to load RAG system: {str(e)}\n{traceback.format_exc()}")

# --- Intent and Subject Detection ---
def detect_user_intent(query):
    query_lower = query.lower()
    intent_patterns = {
        "concept_explanation": [r'explain', r'what is', r'define', r'describe', r'concept of', r'mean'],
        "algorithm_analysis": [r'analyze', r'complexity of', r'big o', r'performance of', r'algorithm for', r'efficiency'],
        "code_debugging": [r'debug', r'error', r'fix', r'problem with', r'incorrect', r'doesn\'t work', r'not working'],
        "exercise_solution": [r'solve', r'solution', r'exercise', r'problem', r'assignment', r'homework'],
        "exam_preparation": [r'prepare', r'exam', r'study', r'review', r'key points', r'important concepts'],
        "project_guidance": [r'project', r'develop', r'implement', r'create', r'build', r'design', r'architecture'],
        "formal_methods": [r'formal', r'proof', r'verify', r'specification', r'correctness', r'z notation', r'schema'],
        "lecture_summary": [r'summarize', r'summary', r'overview', r'lecture', r'course', r'lesson'],
        "exercise_walkthrough": [r'walkthrough', r'step by step', r'solve this', r'work through'],
        "concept_review": [r'review', r'revise', r'revisit', r'understand'],
        "code_explanation": [r'explain code', r'what does this code', r'how does this code']
    }
    intent_scores = {intent: 0 for intent in intent_patterns}
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                intent_scores[intent] += 1
    max_score = max(intent_scores.values())
    if max_score > 0:
        for intent, score in intent_scores.items():
            if score == max_score: return intent
    return "concept_explanation"

def detect_subject_area(query, specified_subject=None):
    if specified_subject and specified_subject in SUBJECT_PROMPTS:
        return specified_subject # Prioritize explicitly passed subject if valid

    query_lower = query.lower()
    subject_scores = {subject: 0 for subject in ESI_TOPICS}
    for subject, keywords in ESI_TOPICS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                subject_scores[subject] += 1
    
    # Check if any of the SUBJECT_PROMPTS keys are in the query
    for subj_key in SUBJECT_PROMPTS.keys():
        if subj_key.lower() in query_lower:
            subject_scores[subj_key] = subject_scores.get(subj_key, 0) + 5 # Boost score for direct match

    max_score = max(subject_scores.values()) if subject_scores else 0
    if max_score > 0:
        for subject, score in subject_scores.items():
            if score == max_score: return subject
    return None

# --- Core Querying Logic ---
def query_esi_rag(query, system_mode="auto", specified_subject_hint=None, language="en"):
    global _loaded_retriever, _loaded_llm
    if not _loaded_retriever or not _loaded_llm:
        load_rag_system() # Ensure system is loaded

    try:
        intent = "concept_explanation"
        subject = None

        if system_mode == "auto":
            intent = detect_user_intent(query)
            subject = detect_subject_area(query, specified_subject_hint)
        elif system_mode in SUBJECT_PROMPTS: # system_mode can be a subject name like "Z Notation"
            subject = system_mode
            intent = detect_user_intent(query) # Still detect intent for Gemini context
        elif system_mode in ESI_PROMPT_TEMPLATES: # system_mode can be an intent name
            intent = system_mode
            subject = detect_subject_area(query, specified_subject_hint)
        
        # Select prompt with better error handling
        selected_prompt = None
        if subject and subject in SUBJECT_PROMPTS:
            selected_prompt = SUBJECT_PROMPTS[subject]
        elif intent and intent in ESI_PROMPT_TEMPLATES:
            selected_prompt = ESI_PROMPT_TEMPLATES[intent]
        else:
            selected_prompt = ESI_PROMPT_TEMPLATES["concept_explanation"] # Fallback
            
        # Ensure we have a valid prompt template
        if selected_prompt is None:
            st.error("Failed to select a prompt template. Using default.")
            selected_prompt = ChatPromptTemplate.from_template('''
            [Context] {context}
            [Question] {input}
            
            Please answer this question based on the provided context and your knowledge.
            ''')
        
        # Create chain and execute
        document_chain = create_stuff_documents_chain(_loaded_llm, selected_prompt)
        current_chain = create_retrieval_chain(_loaded_retriever, document_chain)
        
        # Handle language for the query to Groq if needed (Groq usually infers)
        # For Gemini, language handling is done in get_gemini_response or by prepending to query
        query_for_llm = query
        if language.lower() == "fr":
             # Prepend to query for Gemini, Groq might not need it explicitly
            query_for_llm = f"[Répondre en français s'il vous plaît] {query}"


        result = current_chain.invoke({'input': query_for_llm})
        initial_answer = result['answer']
        
        enhanced_answer = get_gemini_response(
            initial_answer,
            query=query, # Original query for Gemini context
            intent=intent,
            subject=subject
        )
        
        return {
            'raw_response': initial_answer,
            'enhanced_response': enhanced_answer,
            'relevant_documents': result.get('context', []),
            'detected_intent': intent,
            'detected_subject': subject
        }
    except Exception as e:
        raise Exception(f"Error querying ESI RAG system: {str(e)}\n{traceback.format_exc()}")

# --- Z Language Specific Functions (adapted) ---
def query_z_language_rag(query):
    """Query the RAG system specifically for Z language information."""
    # This function now uses query_esi_rag with "Z Notation" as the system_mode/subject
    # to ensure the Z-specific prompt from the dictionary is used.
    return query_esi_rag(query, system_mode="Z Notation", specified_subject_hint="Z Notation")

def extract_code_suggestions(analysis_text, language="text"): # language added for ```language
    try:
        suggestions = []
        # Regex to find code blocks, including optional language specifier
        # Matches ``` optionally followed by a language name, then content, then ```
        code_block_pattern = re.compile(r"```(?:[a-zA-Z0-9_.-]*\n)?(.*?)```", re.DOTALL)
        
        # Find all code blocks in the analysis text
        matches = code_block_pattern.finditer(analysis_text)
        for match in matches:
            # The actual code is in group 1
            code_content = match.group(1).strip()
            if code_content: # Ensure there's content in the block
                suggestions.append(code_content)
        
        # If no formal code blocks found, try a simpler line-based extraction (less reliable)
        if not suggestions:
            lines = analysis_text.split('\n')
            in_code_block = False
            current_suggestion = []
            for line in lines:
                if line.strip().startswith("```") and not in_code_block:
                    in_code_block = True
                    # Skip the opening ``` line itself from being added to suggestion
                    if len(line.strip()) > 3: # handles ```z or ```c
                        pass # language specifier is part of the ``` line
                    continue 
                elif line.strip() == "```" and in_code_block:
                    in_code_block = False
                    if current_suggestion:
                        suggestions.append("\n".join(current_suggestion))
                        current_suggestion = []
                    continue
                if in_code_block:
                    current_suggestion.append(line)
            if current_suggestion: # capture any trailing block
                 suggestions.append("\n".join(current_suggestion))

        return suggestions if suggestions else ["No specific code suggestions extracted."]
    except Exception as e:
        return [f"Error extracting suggestions: {str(e)}"]


def analyze_z_code(z_code):
    """Analyze Z language code for correctness and optimization"""
    try:
        context_result = query_z_language_rag("Z language best practices, common syntax patterns, and schema analysis techniques.")
        
        # Use Gemini to specifically analyze the provided code, with context
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert Z language instructor and formal methods specialist.
        
        Analyze the following Z specification:
        ```z
        {z_code}
        ```
        
        Consider these Z language best practices and common patterns for context:
        {context_result['enhanced_response']}
        
        Provide a detailed analysis covering:
        1. Syntax correctness and adherence to Z notation standards.
        2. Semantic clarity and potential ambiguities.
        3. Type consistency and correctness of declarations.
        4. Soundness of predicates and invariants.
        5. Completeness and appropriateness of operations defined.
        6. Suggestions for improvement, simplification, or alternative formulations, with examples.
        7. Potential errors or common pitfalls evident in the specification.

        Format your response clearly, using markdown for structure. Highlight specific parts of the Z code when discussing them.
        """
        response = model.generate_content(prompt)
        analysis = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)
        
        return {
            'analysis': analysis,
            'suggestions': extract_code_suggestions(analysis, "z")
        }
    except Exception as e:
        raise Exception(f"Error analyzing Z code: {str(e)}")

def generate_optimized_z_code(specification_text):
    """Generate optimized Z language code based on a natural language specification"""
    try:
        context_result = query_z_language_rag("Z language formal specification patterns, schema design principles, and optimization techniques for clarity and rigor.")
        
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
        prompt = f"""
        You are an expert Z language formal specification writer.
        
        Based on this context about Z language:
        {context_result['enhanced_response']}
        
        Generate a formal Z specification for the following requirements:
        "{specification_text}"
        
        Follow these guidelines:
        1. Use proper Z syntax and conventions.
        2. Include helpful comments explaining key sections or complex logic.
        3. Follow best practices for Z programming (e.g., error handling, resource management if applicable).
        4. Make the code efficient and readable.
        5. Use appropriate data structures and algorithms from the standard library if available and suitable.
        
        Return the Z specification code formatted within ```z ... ``` blocks.
        Also, provide a brief explanation of your implementation choices, data structures used, and any important considerations.
        """
        response = model.generate_content(prompt)
        full_response_text = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)

        # Extract Z code from the response
        z_code_match = re.search(r"```z\n(.*?)\n```", full_response_text, re.DOTALL)
        generated_code = z_code_match.group(1).strip() if z_code_match else "No Z specification code block found in the response."
        
        # The rest of the text can be considered an explanation or analysis
        explanation = re.sub(r"```z\n(.*?)\n```", "", full_response_text, flags=re.DOTALL).strip()
        if not explanation and generated_code != "No Z specification code block found in the response.":
            explanation = "Z specification generated as requested."


        return {
            'generated_code': generated_code,
            'analysis': explanation if explanation else "No additional analysis provided.",
            'optimized_suggestions': extract_code_suggestions(explanation, "z") # Suggestions from the explanation part
        }
    except Exception as e:
        raise Exception(f"Error generating Z code: {str(e)}")

# --- ESI Topics and Prompt Dictionaries (from your more advanced backend) ---
ESI_TOPICS = {
    "Algorithms": ["sort", "search", "complexity", "algorithm", "graph", "tree", "dynamic programming", "optimization"],
    "Data Structures": ["array", "list", "tree", "graph", "hash", "stack", "queue", "heap", "binary"],
    "Programming": ["code", "function", "class", "object", "variable", "syntax", "debugging", "compiler"],
    "Computer Architecture": ["processor", "memory", "cache", " 8086 assembly", "cpu", "alu", "register", "instruction"],
    "Operating Systems": ["process", "thread", "memory management", "file system", "scheduling", "deadlock", "concurrency"],
    "Networking": ["protocol", "tcp", "ip", "routing", "socket", "http", "ethernet", "dns", "network", "topology"],
    "Databases": ["sql", "query", "index", "transaction", "normalization", "relational", "schema", "entity"],
    "Software Engineering": ["agile", "requirements", "design", "testing", "uml", "pattern", "architecture", "project"],
    "Formal Methods": ["verification", "specification", "logic", "proof", "invariant", "temporal", "state machine", "z notation"],
    "Artificial Intelligence": ["machine learning", "neural", "natural language", "vision", "clustering", "classification", "reinforcement"],
    "Security": ["encryption", "authentication", "vulnerability", "cryptography", "attack", "firewall", "secure", "cipher"]
}

# Standard prompt templates
ESI_PROMPT_TEMPLATES = {
    "concept_explanation": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

As an ESI educational assistant, explain this computer science concept thoroughly:

1. Provide a clear definition in simple terms
2. Explain the theoretical foundations
3. Show practical applications in computing
4. Connect to related concepts in the ESI curriculum
5. Include code examples when relevant

Format your response with clear headings and bullet points for easy understanding.
'''),

    "algorithm_analysis": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Analyze this algorithm as a computer science instructor would:

1. Explain the algorithm's purpose and key characteristics
2. Break down the steps with detailed explanation
3. Analyze time and space complexity with Big O notation
4. Identify edge cases and limitations
5. Compare with alternative approaches when relevant
6. Show an implementation example in pseudocode or a programming language

Use mathematical notation where appropriate and be precise in your analysis.
'''),

    "code_debugging": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Debug this code as an experienced programming instructor:

1. Identify and explain all errors (syntax, logical, runtime)
2. Provide corrected code with clear comments
3. Explain the underlying issues that caused each error
4. Suggest best practices to avoid similar problems
5. Offer tips for systematic debugging

Show both the incorrect and corrected code for comparison.
'''),

    "exercise_solution": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Solve this ESI course exercise step-by-step:

1. Restate the problem clearly
2. Identify the key concepts and techniques needed
3. Develop a solution strategy
4. Implement the solution with detailed explanations for each step
5. Verify the solution with test cases or examples
6. Discuss alternative approaches if applicable

Show your work thoroughly as would be expected in an ESI course assignment.
'''),

    "exam_preparation": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Help prepare for this ESI exam topic:

1. Summarize the key concepts and theories
2. Outline the most important formulas, algorithms, or methods
3. Provide sample exam questions with detailed solutions
4. Highlight common mistakes to avoid
5. Suggest effective study strategies for this specific topic
6. Connect this topic to others in the curriculum

Present information in an organized manner with clear sections and emphasis on critical points.
'''),

    "project_guidance": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Provide guidance for this ESI project:

1. Break down the project requirements into manageable tasks
2. Suggest a development approach and methodology
3. Recommend technologies, libraries, or tools appropriate for implementation
4. Outline potential challenges and how to address them
5. Provide implementation tips and best practices
6. Suggest testing and validation strategies

Focus on practical, actionable advice that helps advance the project while maintaining educational value.
'''),

    "formal_methods": create_z_language_prompt_template(),
    
    "lecture_summary": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Provide a comprehensive lecture summary based on the following ESI course material:

1. Identify the main topic and subtopics
2. Summarize key concepts, definitions, and theories
3. Extract important formulas, algorithms, or methodologies
4. Highlight practical applications or examples
5. Note connections to other areas in the curriculum
6. Include any exam preparation tips mentioned

Format your response with clear headings, bullet points, and diagrams (described textually) as needed.
'''),

    "exercise_walkthrough": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Provide a detailed step-by-step solution to this ESI exercise:

1. Restate the problem clearly
2. Outline the approach and methodology
3. Work through each step with thorough explanation
4. Include relevant formulas, code snippets, or diagrams as needed
5. Highlight common pitfalls and how to avoid them
6. Provide a final complete solution
7. Add useful tips for solving similar problems

Use clear explanations appropriate for ESI students.
'''),

    "concept_review": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Create a comprehensive review of this computer science concept:

1. Define the concept precisely
2. Explain its theoretical foundations and significance
3. Cover its historical development and key contributors
4. Provide illustrative examples, code implementations, or applications
5. Compare with related or alternative concepts
6. Discuss common misconceptions or challenges in understanding
7. Include relevant equations, diagrams, or algorithms

Structure the response for effective review and exam preparation.
'''),

    "code_explanation": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}

Explain this code with an educational focus:

1. Analyze the code structure and overall purpose
2. Break down the implementation line-by-line
3. Explain key algorithms, data structures, and design patterns used
4. Discuss computational complexity and performance considerations
5. Highlight best practices demonstrated or ways to improve
6. Suggest modifications for different scenarios or requirements
7. Connect to theoretical concepts taught in ESI courses

Use clear annotations and explanations suitable for educational purposes.
''')
}

SUBJECT_PROMPTS = {
    "Z Notation": create_z_language_prompt_template(), # Z-specific prompt for Z Notation subject
    "Assembly Language": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}
Provide an expert explanation on this 8086 Assembly Language topic:
1. Break down the Assembly concepts with clear explanation it is always 8086.
2. Show the relationship between machine code and assembly instructions
3. Explain register usage and memory addressing
4. Provide concrete examples with commented assembly code
5. Compare with high-level language equivalents when helpful
6. Include best practices for efficient assembly programming
Use proper assembly syntax for examples and explain opcodes thoroughly.
'''),
    "C Programming": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}
Explain this C programming concept as an expert instructor:
1. Provide fundamental explanation of the C language feature
2. Show practical code examples with proper memory management
3. Explain pointer usage and memory considerations
4. Discuss common pitfalls and debugging approaches
5. Include optimization techniques when relevant
6. Relate to underlying computer architecture when appropriate
Use standard C syntax and include comments explaining key code sections.
'''),
    "Data Structures": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}
Explain this data structure concept comprehensively:
1. Define the data structure and its primary characteristics
2. Show its internal organization and memory representation
3. Explain the key operations and their time/space complexity
4. Compare with alternative data structures for similar tasks
5. Provide implementation examples in a relevant programming language
6. Discuss real-world applications and use cases
Include diagrams when helpful (describe them textually) and be precise about complexity analysis.
'''),
    "Algorithms": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}
Analyze this algorithm thoroughly:
1. Explain the algorithm's purpose and underlying principles
2. Break down the algorithm steps with clear explanations
3. Analyze time and space complexity with mathematical rigor
4. Discuss edge cases and optimization opportunities
5. Show implementation in pseudocode or an appropriate language
6. Compare with alternative algorithmic approaches
Be precise with complexity notation and provide concrete examples to illustrate the algorithm's behavior.
'''),
    "Computer Architecture": ChatPromptTemplate.from_template('''
[Context] {context}
[Question] {input}
Explain this computer architecture concept in detail:
1. Provide a clear technical explanation of the hardware component or concept
2. Describe how it interacts with other system components
3. Explain its role in the fetch-decode-execute cycle if applicable
4. Discuss performance considerations and design tradeoffs
5. Show how software interfaces with this aspect of architecture
6. Include historical context and evolution when relevant
Use technical terminology precisely and explain concepts at both logical and physical levels when appropriate.
''')
}

# --- Core Querying Logic ---
def query_esi_rag(query, system_mode="auto", specified_subject_hint=None, language="en"):
    global _loaded_retriever, _loaded_llm
    if not _loaded_retriever or not _loaded_llm:
        load_rag_system() # Ensure system is loaded

    try:
        intent = "concept_explanation"
        subject = None

        if system_mode == "auto":
            intent = detect_user_intent(query)
            subject = detect_subject_area(query, specified_subject_hint)
        elif system_mode in SUBJECT_PROMPTS: # system_mode can be a subject name like "Z Notation"
            subject = system_mode
            intent = detect_user_intent(query) # Still detect intent for Gemini context
        elif system_mode in ESI_PROMPT_TEMPLATES: # system_mode can be an intent name
            intent = system_mode
            subject = detect_subject_area(query, specified_subject_hint)
        
        # Select prompt with better error handling
        selected_prompt = None
        if subject and subject in SUBJECT_PROMPTS:
            selected_prompt = SUBJECT_PROMPTS[subject]
        elif intent and intent in ESI_PROMPT_TEMPLATES:
            selected_prompt = ESI_PROMPT_TEMPLATES[intent]
        else:
            selected_prompt = ESI_PROMPT_TEMPLATES["concept_explanation"] # Fallback
            
        # Ensure we have a valid prompt template
        if selected_prompt is None:
            st.error("Failed to select a prompt template. Using default.")
            selected_prompt = ChatPromptTemplate.from_template('''
            [Context] {context}
            [Question] {input}
            
            Please answer this question based on the provided context and your knowledge.
            ''')
        
        # Create chain and execute
        document_chain = create_stuff_documents_chain(_loaded_llm, selected_prompt)
        current_chain = create_retrieval_chain(_loaded_retriever, document_chain)
        
        # Handle language for the query to Groq if needed (Groq usually infers)
        # For Gemini, language handling is done in get_gemini_response or by prepending to query
        query_for_llm = query
        if language.lower() == "fr":
             # Prepend to query for Gemini, Groq might not need it explicitly
            query_for_llm = f"[Répondre en français s'il vous plaît] {query}"


        result = current_chain.invoke({'input': query_for_llm})
        initial_answer = result['answer']
        
        enhanced_answer = get_gemini_response(
            initial_answer,
            query=query, # Original query for Gemini context
            intent=intent,
            subject=subject
        )
        
        return {
            'raw_response': initial_answer,
            'enhanced_response': enhanced_answer,
            'relevant_documents': result.get('context', []),
            'detected_intent': intent,
            'detected_subject': subject
        }
    except Exception as e:
        raise Exception(f"Error querying ESI RAG system: {str(e)}\n{traceback.format_exc()}")

# --- Multi-Agent System Architecture ---
# Agent Roles and System Prompts
AGENT_SYSTEM_PROMPTS = {
    "coordinator": """You are the Coordinator Agent responsible for:
1. Analyzing user queries to determine appropriate retrieval strategy
2. Dispatching tasks to specialized agents
3. Synthesizing their outputs into cohesive responses
4. Managing the overall conversation flow
5. Ensuring high quality, accurate responses through agent validation
""",
    "groq_retriever": """You are the Retrieval Agent powered by Groq's model.
Your responsibilities include:
1. Retrieving relevant information from the vector database
2. Analyzing and understanding the retrieved content
3. Generating initial responses based on the retrieved knowledge
4. Identifying areas where additional information may be needed
5. Providing clear explanations focused on academic quality and accuracy
""",
    "gemini_validator": """You are the Validation Agent powered by Google's Gemini model.
Your responsibilities include:
1. Carefully reviewing responses from the Retrieval Agent
2. Validating factual accuracy of the information
3. Enhancing responses with additional context, examples, or clarifications
4. Ensuring pedagogical quality appropriate for an ESI student audience
5. Restructuring content for clarity, completeness, and correctness
6. Flagging potential inaccuracies or misconceptions for correction
"""
}


if __name__ == "__main__":
    st.set_page_config(page_title="RAG Backend Test Interface", layout="wide")
    st.title("RAG Backend Test Interface")

    if 'system_loaded' not in st.session_state:
        st.session_state.system_loaded = False

    if not st.session_state.system_loaded:
        with st.spinner("Loading RAG system..."):
            try:
                load_rag_system()
                st.session_state.system_loaded = True
                st.success("RAG System Loaded Successfully!")
            except Exception as e:
                st.error(f"Failed to load RAG system: {e}")
                st.stop()
    
    st.header("Test ESI RAG Query")
    test_query = st.text_area("Enter your ESI query:", key="esi_query")
    test_subject_hint = st.text_input("Optional: Subject Hint (e.g., Algorithms, C Programming, Z Notation):", key="esi_subject")
    test_language = st.selectbox("Language:", ["en", "fr"], key="esi_lang")

    if st.button("Submit ESI Query", key="submit_esi"):
        if test_query:
            with st.spinner("Processing ESI query..."):
                try:
                    result = query_esi_rag(test_query, specified_subject_hint=test_subject_hint, language=test_language)
                    st.subheader("Enhanced Response:")
                    st.markdown(result['enhanced_response'])
                    with st.expander("Raw LLM Response"):
                        st.text(result['raw_response'])
                    with st.expander("Detected Intent/Subject"):
                        st.write(f"Intent: {result['detected_intent']}, Subject: {result['detected_subject']}")
                    with st.expander("Relevant Documents"):
                        for doc in result['relevant_documents']:
                            st.text(doc.page_content[:200] + "...")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a query.")

    st.header("Test Z Language Query")
    z_query_test = st.text_area("Enter Z language query:", key="z_query")
    if st.button("Submit Z Query", key="submit_z_q"):
        if z_query_test:
            with st.spinner("Processing Z query..."):
                result = query_z_language_rag(z_query_test)
                st.markdown(result['enhanced_response'])

    st.header("Test Z Code Analysis")
    z_code_test = st.text_area("Enter Z code to analyze:", height=150, key="z_analyze")
    if st.button("Analyze Z Code", key="submit_z_analyze"):
        if z_code_test:
            with st.spinner("Analyzing Z code..."):
                result = analyze_z_code(z_code_test)
                st.markdown("### Analysis:")
                st.markdown(result['analysis'])
                st.markdown("### Suggestions:")
                for s in result['suggestions']:
                    st.code(s, language='z') # Assuming suggestions are Z code or text

    st.header("Test Z Code Generation")
    z_req_test = st.text_area("Enter Z specification requirements:", key="z_generate_req")
    if st.button("Generate Z Code", key="submit_z_generate"):
        if z_req_test:
            with st.spinner("Generating Z code..."):
                result = generate_optimized_z_code(z_req_test)
                st.markdown("### Generated Code:")
                st.code(result['generated_code'], language='z')
                st.markdown("### Analysis/Explanation:")
                st.markdown(result['analysis'])