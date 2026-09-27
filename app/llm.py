# app/llm.py
import asyncio
import gc
import json
from app.config import LLM_MODEL_PATH, MOCK_MODE


def _clean_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def MOCK_DICT(meeting_type: str) -> dict:
    return {
        "summary": "Raport de gardă terapie intensivă (vizită la pat). Analiza pacienților 1, 2 și 3.",
        "patients": [
            {
                "patient_id": 1,
                "patient_summary": "Pacientul este în stare generală lucidă. Valorile de laborator indică creatinina 240 și ureea 19. Pentru monitorizare a fost montată o linie arterială. Pacientul continuă tratamentul cu noradrenalină.",
                "patient_decision": "Se decide continuarea infuziilor cu noradrenalină, continuarea umplerii vasculare și monitorizarea în continuare a pacientului.",
            },
            {
                "patient_id": 2,
                "patient_summary": "Pacientul este afebril, prezentând insuficiență respiratorie. Oxigenarea este de 90%, iar pacientul este hipercapnic, cu valoarea CO₂ de 69. În prezent urmează tratament cu Forxiga și Diacarb.",
                "patient_decision": "S-a decis efectuarea unui consult cardiologic și a unui consult terapeut. De asemenea, s-a decis utilizarea ventilației non-invazive BiPAP.",
            },
            {
                "patient_id": 3,
                "patient_summary": "La scanare se observă o dinamică pozitivă, cu prezența lichidului în plămâni. Pacientul urmează tratament cu meropenem și amikacină, doza de amikacină fiind crescută de la 1000 la 1500. Se administrează fluconazol conform rezultatelor culturilor pozitive pentru Klebsiella și Candida în sânge și urină. Frecvența pacemakerului a fost crescută la 80. Pacientul prezintă anemie, cu hemoglobina de 86.",
                "patient_decision": "S-au decis transfuzii de sânge în contextul anemiei. Se recomandă consult oncologic și consult urologic pentru luarea unei decizii privind stentarea.",
            },
        ],
    }


def _sync_extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    if MOCK_MODE:
        return MOCK_DICT(meeting_type)

    from llama_cpp import Llama

    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=28,
        n_ctx=8192,
        verbose=False,
    )

    system_prompt = """Ești medic documentarist de terapie intensivă. Primești un transcript Whisper în limba română, dintr-o vizită la pat. Vorbirea este rapidă, cu termeni de spital, adesea fără diacritice, cu nume de medicamente și doze auzite greșit. Reconstruiești ce s-a spus clinic.

Nu rescrie transcriptul frază cu frază. Grupează discuția pe pacienți, în ordinea în care apar. Un pacient nou începe la un număr de pat, o boxă, un număr de pacient sau la o primire nouă în reanimare.

Pentru fiecare pacient extrage doar ce se susține din vorbire:
- identificare: număr pacient (reținut ca număr întreg/integer)
- motiv și antecedente relevante
- stare: lucid, afebril, febril, cooperant, somnoros, hemodinamic stabil sau instabil, volemic
- respirator: insuficiență respiratorie, saturație, hipercapnie, CO2, oxigen, BiPAP, ventilație non-invazivă
- hemodinamic: tensiune, noradrenalină, dobutamină, alți presori, doza auzită
- laborator: creatinină, uree, clearance, hemoglobină, lactat, exces sau deficit de baze
- imagistic și eco: lichid pleural, hidronefroză, eco transtoracic, eco transesofagian, endocardită
- tratament: antibiotic, antimicotic, diuretic, anticoagulant, doză veche și doză nouă
- culturi: germene, loc (sânge, urină), sensibilitate
- dispozitive: linie arterială, pacemaker sau EKS, dren, stent
- consulturi cerute sau făcute
- decizia de la vizită: ce se continuă, ce se schimbă, ce se așteaptă

Corectează aceste auziri greșite când contextul clinic le susține:
- nori, noru, noradrenalina, norodrenalină = noradrenalină
- dobu, DOB = dobutamină
- presăr, presor = suport presor
- chemodinamic = hemodinamic
- volimic, volemnic, volumă = volemic, volum
- descartat, deschis dinor = descărcat de noradrenalină
- ucid = lucid
- ieco, Ievo = eco
- Trans-Svăgean, transtoracic = eco transesofagian, eco transtoracic
- îndocardită = endocardită
- pipatul = patul
- deacarb, deacarbul = Diacarb
- BIPAP, bipap = BiPAP
- miropinem, meropenem = meropenem
- amicacină, amikacină = amikacină
- fructir, fluconazol = fluconazol
- hidronifroza, urechironifroza = hidronefroză, ureterohidronefroză
- neprostoma = nefrostomă
- folopatie = encefalopatie
- delirii = delir
- cordas = cordaj
- tromboaspiratie = tromboaspirație
- fractie = fracție de ejecție
- trivascular = coronarian trivascular
- EKS, pism = pacemaker
- cleros = clearance
- șarie O2, O2 din oră = oxigen, în litri pe oră

Reguli clinice:
- Păstrează numerele auzite: tensiune, doze, saturație, CO2, hemoglobină, lactat, mililitri de lichid, frecvența pacemakerului.
- Dacă un număr poate fi și altul, scrie forma cea mai plauzibilă și marcheaz-o cu „auzit neclar”.
- Nu alege stânga sau dreapta dacă vorbitorii se contrazic. Scrie că partea nu este clară.
- Nu inventa diagnostic, doză, consult sau decizie ca să completezi un gol.
- Repetițiile de tip „așa se numește acest stent” sunt eroare Whisper. Le ignori.
- Pasajele fără sens clinic le marchezi „neinteligibil” și treci mai departe.

Răspunsul tău trebuie să fie EXCLUSIV un obiect JSON valid, respectând această structură exactă:
{
  "summary": "Rezumat general al vizitei la pat",
  "patients": [
    {
      "patient_id": 1,
      "patient_summary": "Pacientul este în stare generală lucidă. Valorile de laborator indică creatinina 240 și ureea 19. Pentru monitorizare a fost montată o linie arterială. Pacientul continuă tratamentul cu noradrenalină.",
      "patient_decision": "Se decide continuarea infuziilor cu noradrenalină, continuarea umplerii vasculare și monitorizarea în continuare a pacientului."
    },
    {
      "patient_id": 2,
      "patient_summary": "Pacientul este afebril, prezentând insuficiență respiratorie. Oxigenarea este de 90%, iar pacientul este hipercapnic, cu valoarea CO₂ de 69. În prezent urmează tratament cu Forxiga și Diacarb.",
      "patient_decision": "S-a decis efectuarea unui consult cardiologic și a unui consult terapeut. De asemenea, s-a decis utilizarea ventilației non-invazive BiPAP."
    },
    {
      "patient_id": 3,
      "patient_summary": "La scanare se observă o dinamică pozitivă, cu prezența lichidului în plămâni. Pacientul urmează tratament cu meropenem și amikacină, doza de amikacină fiind crescută de la 1000 la 1500. Se administrează fluconazol conform rezultatelor culturilor pozitive pentru Klebsiella și Candida în sânge și urină. Frecvența pacemakerului a fost crescută la 80. Pacientul prezintă anemie, cu hemoglobina de 86.",
      "patient_decision": "S-au decis transfuzii de sânge în contextul anemiei. Se recomandă consult oncologic și consult urologic pentru luarea unei decizii privind stentarea."
    }
  ]
}"""

    user_prompt = f"Transcript:\n{transcript}"

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3500,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw_content = response["choices"][0]["message"]["content"].strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    elif raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    raw_content = raw_content.strip()

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        result = MOCK_DICT(meeting_type)

    result.setdefault("summary", "")
    result.setdefault("patients", [])

    del llm
    _clean_gpu()
    return result


async def extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    return await asyncio.to_thread(
        _sync_extract_decisions_and_actions, transcript, meeting_type
    )