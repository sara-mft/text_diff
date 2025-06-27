import requests
from bs4 import BeautifulSoup
import openai
import json
import time
from pathlib import Path
from html import escape

# ========== CONFIGURATION ==========
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
AZURE_DEPLOYMENT_NAME = "gpt-4o" 

HEADERS = {"api-key": AZURE_OPENAI_KEY}
#MODEL_URL = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
MODEL_URL = AZURE_OPENAI_ENDPOINT

REFERENCE_TEXT_PATH = "Reference.txt" 
TARGET_URLS = [
    "https://www.caisse-epargne.fr/environnement-solutions-durables-responsables/",
    "https://www.banquepopulaire.fr/sud/banque-de-la-transition-energetique/",
    "https://www.banquepopulaire.fr/conseils/agir-developpement-durable-isr/",
    "https://www.palatine.fr/clients-prives/notre-offre-verte/",
]

# ========== FUNCTIONS ==========

def extract_clean_text(url):
    print(f"Scraping: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36",
        "Accept-Language": "fr-FR,fr;q=0.9",
    }

    response = requests.get(url, headers=headers)
    
    #response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        script.extract()

    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def load_reference_text(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def query_gpt4(reference, website_text):
    prompt = f"""
    Vous êtes un expert analyste chargé d’évaluer la mise en œuvre du développement durable par une banque.

    Comparez le contenu ci-dessous (provenant d’un **site web**) avec le **texte de référence**.

    ### TEXTE DE RÉFÉRENCE :
    {reference}

    ### CONTENU DU SITE WEB :
    {website_text}

    ### QUESTION :
    Le site web contient-il des informations et/ou des produits *supplémentaires ou uniques* qui ne sont **pas déjà présentes** dans le texte de référence ? Si oui, listez-les ou résumez-les clairement.

    Retournez la réponse au format Markdown avec un en-tête : `## Résultats de la comparaison` suivi de points de liste pour chaque élément nouveau.
    """

    payload = {
        "messages": [
            {"role": "system", "content": "You are a sustainability analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return prompt, response.json()["choices"][0]["message"]["content"]


def generate_outputs(results):
    Path("outputs3").mkdir(exist_ok=True)

    # .txt file
    with open("outputs3/comparaison_resultats.txt", "w", encoding="utf-8") as f_txt:
        for r in results:
            f_txt.write(f"URL : {r['url']}\n")
            f_txt.write("\n--- PROMPT ---\n")
            f_txt.write(r['prompt'] + "\n")
            f_txt.write("\n--- RÉPONSE GPT-4 ---\n")
            f_txt.write(r['response'] + "\n")
            f_txt.write("\n" + "="*80 + "\n")

    # HTML file
    html_parts = [
        "<!DOCTYPE html><html lang='fr'><head><meta charset='UTF-8'>",
        "<title>Comparaison GPT-4 - Sites Web vs Référence BPCE</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#f4f4f4;padding:2em;max-width:1000px;margin:auto;}",
        "h1{color:#222;} h2{color:#004080;} pre{background:#fff;padding:1em;border:1px solid #ccc;border-radius:8px;white-space:pre-wrap;word-wrap:break-word;}",
        "details{margin-bottom:2em;} summary{font-weight:bold;cursor:pointer;}",
        "a{color:#007acc;text-decoration:none;} a:hover{text-decoration:underline;}",
        "</style></head><body>",
        "<h1>📊 Rapport comparatif GPT-4 : Sites Web vs Référence</h1>"
    ]

    for r in results:
        html_parts.append(f"<h2>🔗 <a href='{r['url']}' target='_blank'>{r['url']}</a></h2>")
        html_parts.append(f"<h3>✅ Résumé de la réponse GPT-4</h3>")
        html_parts.append(f"<pre>{escape(r['response'])}</pre>")
        html_parts.append("<details><summary>📝 Voir le prompt complet</summary>")
        html_parts.append(f"<pre>{escape(r['prompt'])}</pre>")
        html_parts.append("</details>")

    html_parts.append("</body></html>")

    with open("outputs3/comparaison_resultats.html", "w", encoding="utf-8") as f_html:
        f_html.write("\n".join(html_parts))


def main():
    reference = load_reference_text(REFERENCE_TEXT_PATH)
    results = []

    for url in TARGET_URLS:
        print("*********************************")
        print(url)
        try:
            website_text = extract_clean_text(url)
            prompt, comparison = query_gpt4(reference, website_text)
            results.append({
                "url": url,
                "prompt": prompt,
                "response": comparison
            })
            time.sleep(1.5)
        except Exception as e:
            results.append({
                "url": url,
                "prompt": "",
                "response": f"Error: {str(e)}"
            })

    # Save as original markdown
    with open("outputs3/comparison_report.md", "w", encoding="utf-8") as file:
        for r in results:
            file.write(f"# Report for {r['url']}\n")
            file.write(r['response'] + "\n\n")

    # Save as JSON
    with open("outputs3/comparison_report.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    generate_outputs(results)

    print("✅ Résultats enregistrés : 'comparison_report.md', 'comparison_report.json', 'outputs/comparaison_resultats.txt' et 'outputs/comparaison_resultats.html'")

