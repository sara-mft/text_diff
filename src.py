import difflib
import pandas as pd

import string

def clean_text(text):
    """Convert text to lowercase and remove punctuation."""
    if isinstance(text, str):
        return text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

def highlight_character_diff(text1, text2):
    matcher = difflib.SequenceMatcher(None, text1, text2)
    highlighted = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":  
            highlighted.append(f'<span style="color: red; font-weight: bold;">{text1[i1:i2]}</span>')
            highlighted.append(f'<span style="color: green; font-weight: bold;">{text2[j1:j2]}</span>')
        elif tag == "delete":  
            highlighted.append(f'<span style="color: red; font-weight: bold;">{text1[i1:i2]}</span>')
        elif tag == "insert":  
            highlighted.append(f'<span style="color: green; font-weight: bold;">{text2[j1:j2]}</span>')
        else: 
            highlighted.append(text1[i1:i2])
    
    return "".join(highlighted)

if __name__ == "__main__":

    data = {
        "Column1": ["good morning", "this is a test"],
        "Column2": ["good evening", "this is an exam"]
    }
    df = pd.DataFrame(data)

    df = df.applymap(clean_text)
    
    df["Difference"] = df.apply(lambda row: highlight_character_diff(row["Column1"], row["Column2"]), axis=1)
    
    html_content = f'''<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DataFrame Column Differences</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f4f4f4; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            span {{ padding: 2px 4px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h2>DataFrame Column Differences</h2>
        {df.to_html(escape=False, index=False)}
    </body>
    </html>'''
    
    # Écrire dans un fichier
    with open("diff_output.html", "w", encoding="utf-8") as file:
        file.write(html_content)
    
    print("Diff HTML generated: diff_output.html")
