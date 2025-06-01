from flask import Flask, render_template, request, send_file
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from fpdf import FPDF
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
nlp = spacy.load("pt_core_news_sm") 

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def preprocessar_texto(texto):
    termos_tecnicos = {
        'clorose': 'clorose',
        'clorótico': 'clorose',
        'deformação': 'deformação',
        'enrolamento': 'enrolamento',
        'necrosamento': 'necrose',
        'tortuosidade': 'deformação',
        'míldio': 'míldio',
        'verticilium': 'verticilium'
    }
    doc = nlp(texto.lower())
    lemas = []
    for token in doc:
        if token.text in termos_tecnicos:
            lemas.append(termos_tecnicos[token.text])
        elif not token.is_stop and not token.is_punct and not token.like_num:
            lemas.append(token.lemma_)
    return " ".join(lemas)

def formatar_markdown(texto):
    texto = re.sub(r"### (.*?)\n", r"<h3>\1</h3>\n", texto)  # Título
    texto = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", texto)  # Negrito
    texto = re.sub(r"\- (.*?)\n", r"• \1<br>", texto)  # Lista
    texto = texto.replace('\n', '<br>')
    return texto

df = pd.read_csv("doencas_algodoeiro.csv")
df.dropna(inplace=True)

dados_extra = {
    'Doença': ['Clorose Ferrica', 'Tombamento', 'Mancha Angular', 'Míldio', 'Vírus do Enrolamento'],
    'Características': [
        'folhas jovens com clorose internerval e deformações',
        'plantas jovens tombando com lesões escuras no colo',
        'manchas angulares com bordas aquosas e centro necrosado',
        'manchas esbranquiçadas na face inferior das folhas',
        'enrolamento foliar com espessamento das nervuras'
    ],
    'Descrição': [
        'Deficiência de ferro causando amarelecimento entre nervuras',
        'Doença de solo que afeta plântulas',
        'Doença bacteriana comum em condições úmidas',
        'Doença fúngica que forma esporos brancos',
        'Doença viral transmitida por insetos'
    ]
}
df_extra = pd.DataFrame(dados_extra)
df = pd.concat([df, df_extra], ignore_index=True)
df["Características"] = df["Características"].astype(str).apply(preprocessar_texto)

X_doenca = df["Características"]
y_doenca = df["Doença"]
modelo_doenca = make_pipeline(TfidfVectorizer(), MultinomialNB())
modelo_doenca.fit(X_doenca, y_doenca)

entradas_filtro = list(X_doenca) + [
    "qual é a capital da bahia", "vou te bater", "quanto é 2 + 2", "como está o tempo",
    "meu cachorro está doente", "pipoca salgada", "qual sua idade", "filme bom",
    "música brasileira", "notícias de hoje", "celular novo", "computador rápido"
]
rotulos_filtro = ["relevante"] * len(X_doenca) + ["irrelevante"] * (len(entradas_filtro) - len(X_doenca))
modelo_filtro = make_pipeline(TfidfVectorizer(), MultinomialNB())
modelo_filtro.fit(entradas_filtro, rotulos_filtro)

def verificar_relevancia(texto):
    entrada_limpa = preprocessar_texto(texto)
    tokens = entrada_limpa.split()
    palavras_chave_algodao = [
        "folha", "mancha", "fungo", "bactéria", "vírus", "praga", "inseto",
        "descolor", "amarel", "murch", "sec", "podr", "lesão", "crescimento",
        "algodoeiro", "planta", "cultivo", "raiz", "caule", "sintoma",
        "doença", "infect", "patógeno", "necrose", "clorose", "enrug", "desfolha",
        "deformação", "tortuosidade", "enrolamento", "distorção", "clorótico",
        "amarelecimento", "necrosamento", "míldio", "ferrugem", "verticilium",
        "tombamento", "angular", "aquoso", "nervura", "esporo", "plântula",
        "colo", "internerval", "esbranquiçado", "viral", "fúngico", "bacteriano"
    ]
    palavras_irrelevantes = [
        "futebol", "jogar", "bola", "filme", "música", "receita", "comida",
        "carro", "celular", "computador", "notícia", "clima", "tempo", "animal"
    ]
    count_relevante = sum(1 for token in tokens if any(p in token for p in palavras_chave_algodao))
    count_irrelevante = sum(1 for token in tokens if any(p in token for p in palavras_irrelevantes))
    contexto = modelo_filtro.predict([entrada_limpa])[0]
    if (len(tokens) >= 3 and count_relevante >= 2 and count_relevante >= (2 * count_irrelevante) and contexto == "relevante"):
        return True
    return False

def obter_recomendacao_combate(doenca):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "HTTP-Referer": "https://your-site-url.com",
        "X-Title": "Cotton Disease Assistant"
    }
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "Você é um especialista em agricultura. Forneça recomendações práticas e diretas para combater doenças do algodoeiro."},
            {"role": "user", "content": f"Como combater {doenca} no algodoeiro?"}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erro ao obter recomendação: {str(e)}"

def gerar_relatorio(doenca):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "HTTP-Referer": "https://your-site-url.com",
        "X-Title": "Cotton Disease Assistant"
    }
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": (
                "Você é um pesquisador acadêmico. Acesse e consulte bases científicas como SciELO, PubMed, Google Scholar. "
                "Crie um relatório técnico original de 300 a 500 palavras sobre doenças do algodoeiro. Evite copiar literalmente. "
                "Cite as fontes no final, se possível."
            )},
            {"role": "user", "content": f"Crie um relatório técnico sobre {doenca} no algodoeiro."}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Não foi possível gerar o relatório técnico. Erro: {str(e)}"

def gerar_pdf(conteudo, doenca):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_fill_color(57, 106, 177)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, f"RELATÓRIO TÉCNICO: {doenca.upper()}", ln=1, fill=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", ln=1, align='R')
    pdf.ln(10)
    for paragraph in conteudo.split('\n\n'):
        partes = re.split(r"(\*\*.*?\*\*)", paragraph)
        for parte in partes:
            if parte.startswith("**") and parte.endswith("**"):
                texto_negrito = parte[2:-2]
                pdf.set_font("Arial", 'B', 12)
                pdf.multi_cell(0, 6, texto_negrito)
                pdf.set_font("Arial", '', 12)
            else:
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 6, parte)
        pdf.ln(5)
    pdf.set_y(-15)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Relatório gerado pelo Sistema de Diagnóstico de Doenças do Algodoeiro", 0, 0, 'C')
    nome = f"relatorio_{uuid.uuid4().hex}.pdf"
    caminho = os.path.join("relatorios", nome)
    os.makedirs("relatorios", exist_ok=True)
    pdf.output(caminho)
    return caminho

def prever_doenca(texto):
    texto_original = texto.lower().strip()
    if texto_original == "sair":
        return "sair"
    if re.fullmatch(r"[0-9\s]+", texto_original) or len(texto_original) < 4:
        return "Entrada inválida. Descreva os sintomas."
    if not verificar_relevancia(texto_original):
        return "Descreva apenas sintomas da planta para diagnóstico."
    texto_processado = preprocessar_texto(texto_original)
    if len(texto_processado.split()) < 3:
        return "Descreva melhor os sintomas (mínimo 3 palavras)."
    try:
        doenca = modelo_doenca.predict([texto_processado])[0]
        descricao = df[df["Doença"] == doenca]["Descrição"].values[0]
        descricao_html = descricao.replace("\n", "<br>")
        recomendacao = formatar_markdown(obter_recomendacao_combate(doenca))
        relatorio = gerar_relatorio(doenca)
        pdf_path = gerar_pdf(relatorio, doenca)
        return f"""
<div class=\"resposta-container\">
    <div class=\"resposta-bloco\">
        <h2>Doença Identificada</h2>
        <p class=\"resposta-doenca\">{doenca}</p>
    </div>
    <div class=\"resposta-bloco\">
        <h3>Descrição</h3>
        <p>{descricao_html}</p>
    </div>
    <div class=\"resposta-bloco\">
        <h3>Recomendações de Combate</h3>
        <p>{recomendacao}</p>
    </div>
    <div class=\"resposta-bloco\">
        <h3>Relatório Técnico</h3>
        <a class=\"botao-relatorio\" href=\"/download/{os.path.basename(pdf_path)}\" target=\"_blank\">Baixar PDF</a>
    </div>
</div>
"""
    except Exception as e:
        return f"Erro ao processar os sintomas: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prever", methods=["POST"])
def prever():
    texto = request.form.get("sintomas")
    resposta = prever_doenca(texto)
    return f'<div id="resultado">{resposta}</div>'

@app.route("/download/<filename>")
def download(filename):
    caminho = os.path.join("relatorios", filename)
    return send_file(caminho, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
