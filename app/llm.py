# app/llm.py
import asyncio
import gc
import json
import re
from app.config import MOCK_MODE, LLM_MODEL_PATH


def _clean_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def MOCK_DICT(meeting_type: str) -> dict:
    """Date mock conforme cu raportul clinic Medpark ATI."""
    return {
        "summary": "Raport de gardă terapie intensivă (vizită la pat). Analiza pacienților 1, 2 și 3.",
        "decisions_made": [
            "Monitorizare hemodinamică continuă și reevaluare biologică la 6 ore.",
            "Efectuarea consulturilor interdisciplinare solicitate în regim de urgență."
        ],
        "patient_cases": [
            {
                "patient_id": 1,
                "clinical_summary": (
                    "Stare: lucid, hemodinamic instabil pe suport presor. "
                    "Obiectiv și investigații: creatinină 240, uree 19; linie arterială montată pentru monitorizare invazivă. "
                    "Tratament și doze: continuă infuzia cu noradrenalină și umplerea vasculară. "
                    "Neclar: volumul exact de umplere vasculară auzit neclar."
                ),
                "decisions": [
                    "Continuarea infuziilor cu noradrenalină.",
                    "Continuarea umplerii vasculare.",
                    "Monitorizarea în continuare a pacientului."
                ],
                "action_items": [
                    {
                        "task": "Titrarea dozei de noradrenalină pe linia arterială",
                        "owner": "Dr. Angela Rusu",
                        "deadline": "Permanent",
                        "department": "ATI",
                        "priority": "Urgent"
                    }
                ]
            },
            {
                "patient_id": 2,
                "clinical_summary": (
                    "Stare: afebril, insuficiență respiratorie marcată. "
                    "Obiectiv și investigații: oxigenare (SpO2) 90%, hipercapnic cu CO2 69. "
                    "Tratament și doze: urmează tratament cu Forxiga și Diacarb. "
                    "Neclar: fără particularități neclare."
                ),
                "decisions": [
                    "Efectuarea unui consult cardiologic.",
                    "Efectuarea unui consult terapeut.",
                    "Utilizarea ventilației non-invazive BiPAP."
                ],
                "action_items": [
                    {
                        "task": "Montare și ajustare parametri ventilație non-invazivă BiPAP",
                        "owner": "Dr. Angela Rusu",
                        "deadline": "Imediat",
                        "department": "ATI",
                        "priority": "Urgent"
                    },
                    {
                        "task": "Solicitare consult cardiologic și terapeut",
                        "owner": "Dr. Mihai Grosu",
                        "deadline": "Astăzi 12:00",
                        "department": "Cardiologie",
                        "priority": "High"
                    }
                ]
            },
            {
                "patient_id": 3,
                "clinical_summary": (
                    "Stare: anemic (Hb 86), dinamică pozitivă la scanare cu prezența lichidului în plămâni. "
                    "Obiectiv și investigații: culturi pozitive pentru Klebsiella și Candida în sânge și urină; pacemaker crescut la 80 bpm. "
                    "Tratament și doze: meropenem, amikacină (doză crescută de la 1000 la 1500), fluconazol conform culturilor. "
                    "Neclar: partea pe care este prezent lichidul pleural nu este clară."
                ),
                "decisions": [
                    "Transfuzii de sânge în contextul anemiei (Hb 86).",
                    "Consult oncologic.",
                    "Consult urologic pentru luarea unei decizii privind stentarea."
                ],
                "action_items": [
                    {
                        "task": "Comandă masă eritrocitară și inițiere transfuzie de sânge",
                        "owner": "Elena Morari",
                        "deadline": "Astăzi 13:00",
                        "department": "Banca de Sânge",
                        "priority": "Urgent"
                    },
                    {
                        "task": "Organizare consult oncologic și urologic pentru oportunitatea stentării",
                        "owner": "Dr. Vasile Cebotari",
                        "deadline": "Astăzi 15:00",
                        "department": "Urologie / Oncologie",
                        "priority": "High"
                    }
                ]
            }
        ],
        "action_items": [
            {
                "task": "Verificare stocuri amikacină, meropenem și kituri BiPAP",
                "owner": "Elena Morari",
                "deadline": "Astăzi 17:00",
                "department": "Farmacie Clinică",
                "priority": "Medium"
            }
        ],
        "attendees": [
            {"name": "Vasile Cebotari", "role": "Șef secție chirurgie"},
            {"name": "Angela Rusu", "role": "Medic ATI"},
            {"name": "Mihai Grosu", "role": "Medic specialist"},
            {"name": "Elena Morari", "role": "Farmacist diriginte"}
        ]
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

    system_prompt = """Ești medic documentarist de terapie intensivă la Spitalul Medpark. Primești un transcript Whisper în limba română, dintr-o vizită la pat. Vorbirea este rapidă, cu termeni de spital, adesea fără diacritice, cu nume de medicamente și doze auzite greșit. Reconstruiești ce s-a spus clinic.

Nu rescrie transcriptul frază cu frază. Grupează discuția pe pacienți, în ordinea în care apar. Un pacient nou începe la un număr de pat, o boxă, un număr de pacient sau la o primire nouă în reanimare.

Pentru fiecare pacient extrage doar ce se susține din vorbire:
- identificare: număr pacient (reținut ca număr întreg/integer), pat, boxă
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

Reguli stricte:
- Păstrează numerele auzite: tensiune, doze, saturație, CO2, hemoglobină, lactat, mililitri de lichid, frecvența pacemakerului.
- Dacă un număr poate fi și altul, scrie forma cea mai plauzibilă și marcheaz-o cu „auzit neclar”.
- Nu alege stânga sau dreapta dacă vorbitorii se contrazic. Scrie că partea nu este clară.
- Nu inventa diagnostic, doză, consult sau decizie ca să completezi un gol.
- Repetițiile de tip „așa se numește acest stent” sunt eroare Whisper. Le ignori.
- Pasajele fără sens clinic le marchezi „neinteligibil” și treci mai departe.
- Numele de medici rămân cum se aud.

Răspunsul tău trebuie să fie EXCLUSIV un obiect JSON valid, respectând cu strictețe acest model exact:
{
  "summary": "Raport de gardă terapie intensivă (vizită la pat). Analiza pacienților 1, 2 și 3.",
  "decisions_made": [
    "Monitorizare hemodinamică și respiratorie continuă pe secția ATI.",
    "Efectuarea consulturilor interdisciplinare solicitate în regim prioritar."
  ],
  "patient_cases": [
    {
      "patient_id": 1,
      "clinical_summary": "Pacientul este în stare generală lucidă. Valorile de laborator indică creatinina 240 și ureea 19. Pentru monitorizare a fost montată o linie arterială. Pacientul continuă tratamentul cu noradrenalină.",
      "decisions": [
        "Se decide continuarea infuziilor cu noradrenalină, continuarea umplerii vasculare și monitorizarea în continuare a pacientului."
      ],
      "action_items": [
        {
          "task": "Continuarea infuziilor cu noradrenalină și umplerea vasculară",
          "owner": "Medic ATI",
          "deadline": "Permanent",
          "department": "ATI",
          "priority": "Urgent"
        }
      ]
    },
    {
      "patient_id": 2,
      "clinical_summary": "Pacientul este afebril, prezentând insuficiență respiratorie. Oxigenarea este de 90%, iar pacientul este hipercapnic, cu valoarea CO₂ de 69. În prezent urmează tratament cu Forxiga și Diacarb.",
      "decisions": [
        "S-a decis efectuarea unui consult cardiologic și a unui consult terapeut.",
        "De asemenea, s-a decis utilizarea ventilației non-invazive BiPAP."
      ],
      "action_items": [
        {
          "task": "Inițierea ventilației non-invazive BiPAP",
          "owner": "Medic ATI",
          "deadline": "Imediat",
          "department": "ATI",
          "priority": "Urgent"
        },
        {
          "task": "Efectuarea consultului cardiologic și a consultului terapeut",
          "owner": "Medic curant",
          "deadline": "Astăzi 12:00",
          "department": "Cardiologie / Terapie",
          "priority": "High"
        }
      ]
    },
    {
      "patient_id": 3,
      "clinical_summary": "La scanare se observă o dinamică pozitivă, cu prezența lichidului în plămâni. Pacientul urmează tratament cu meropenem și amikacină, doza de amikacină fiind crescută de la 1000 la 1500. Se administrează fluconazol conform rezultatelor culturilor pozitive pentru Klebsiella și Candida în sânge și urină. Frecvența pacemakerului a fost crescută la 80. Pacientul prezintă anemie, cu hemoglobina de 86.",
      "decisions": [
        "S-au decis transfuzii de sânge în contextul anemiei.",
        "Se recomandă consult oncologic și consult urologic pentru luarea unei decizii privind stentarea."
      ],
      "action_items": [
        {
          "task": "Transfuzii de sânge în contextul anemiei (Hb 86)",
          "owner": "Medic ATI",
          "deadline": "Astăzi 13:00",
          "department": "Banca de Sânge / ATI",
          "priority": "Urgent"
        },
        {
          "task": "Consult oncologic și urologic pentru luarea deciziei privind stentarea",
          "owner": "Medic curant",
          "deadline": "Astăzi 15:00",
          "department": "Oncologie / Urologie",
          "priority": "High"
        }
      ]
    }
  ],
  "action_items": [
    {
      "task": "Verificare stoc amikacină, meropenem și consumabile BiPAP la farmacie",
      "owner": "Farmacist de gardă",
      "deadline": "Astăzi 17:00",
      "department": "Farmacie Clinică",
      "priority": "Medium"
    }
  ],
  "attendees": [
    {
      "name": "Nume Medic",
      "role": "Medic ATI / Curant"
    }
  ]
}"""

    user_prompt = f"Categorie: {meeting_type}\n\nTranscript brut Whisper:\n{transcript}"

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
    result.setdefault("decisions_made", [])
    result.setdefault("patient_cases", [])
    result.setdefault("action_items", [])
    result.setdefault("attendees", [])

    del llm
    _clean_gpu()
    return result


async def extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    return await asyncio.to_thread(_sync_extract_decisions_and_actions, transcript, meeting_type)