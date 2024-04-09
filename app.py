# %% Définition des imports
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import pandas as pd
from langchain.chains.llm import LLMChain
from langchain.evaluation import JsonEditDistanceEvaluator, JsonSchemaEvaluator
from langchain.globals import set_verbose
from langchain.model_laboratory import ModelLaboratory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pouvoir afficher les logs de langchain. Utile pour voir l'intégralité du prompt.
set_verbose(False)


# %% Définition des fonctions utiles
def iterate_benchmark_files(
    folder_path: Path,
) -> Generator[Tuple[str, "DocumentLawArticles", str], None, None]:
    """Itère sur les fichiers de benchmark et retourne le contenu des fichiers json et txt associés.
    - Le fichier json contient les articles de lois de référence et
    - Le fichier txt contient le texte brut du document.
    """
    for file_path in Path(folder_path).glob("*.txt"):
        yield (
            file_path.stem,
            DocumentLawArticles.parse_file(file_path.with_suffix(".json")),
            file_path.read_text(),
        )


def normalize_output(json_str: str) -> str:
    """Normalise la sortie des modèles pour la comparer plus facilement."""
    data = json.loads(json_str)
    data["articles"] = sorted(
        set(tuple(sorted(d.items())) for d in data["articles"]),
        key=lambda x: dict(x)["num"],
    )
    data["articles"] = [dict(t) for t in data["articles"]]
    return json.dumps(data, separators=(",", ":")).encode().decode("unicode_escape")


def get_code_list() -> List[str]:
    """Récupère la liste des codes de lois disponibles sur legifrance."""
    codes = []
    with open("./data/legifrance/codes.json", "r") as file:
        for code in json.load(file)["results"]:
            codes.append(code["titre"])
    return codes


def get_model_name(llm: BaseChatModel) -> str:
    """Hack: Récupère le nom du modèle utilisé. On constate un problème de normalistation des attributs entre les différente classes de modèles."""
    if hasattr(llm, "model"):
        return llm.model
    return llm.model_name


class LawArticle(BaseModel):
    """Modèle pour un article de loi et son code associé."""

    num: str = Field(title="num", description="numéro de l'article de loi")
    code: Optional[str] = Field(
        title="code", description="code de l'article s'il est précisé"
    )

    @validator("num", pre=True)
    def transform_num_to_str(cls, value) -> str:
        return str(value).replace(" ", "").replace(".", "")

    @validator("code", pre=True)
    def title_code(cls, value) -> str:
        return value.capitalize()


class DocumentLawArticles(BaseModel):
    articles: Optional[list[LawArticle]] = None


# %% Définition du prompt, des modèles à tester et d'un parser en sortie.
output_parser = PydanticOutputParser(pydantic_object=DocumentLawArticles)
prompt = PromptTemplate.from_template(
    Path("./system_prompt.md").read_text(),
    partial_variables={
        "output_format": output_parser.get_format_instructions(),
        "codes": "\n".join(get_code_list()),
    },
)

llms = [
    ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
    ChatAnthropic(model="claude-3-haiku-20240307", temperature=0),
    ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0),
    ChatAnthropic(model="claude-3-opus-20240229", temperature=0),
    ChatMistralAI(model="mistral-small-latest", temperature=0),
    ChatMistralAI(model="mistral-medium-latest", temperature=0),
    ChatMistralAI(model="mistral-large-latest", temperature=0),
]

chains = [LLMChain(llm=llm, prompt=prompt) for llm in llms]


# %% Benchmark des modèles
# Evaluators:
json_distance_evaluator = JsonEditDistanceEvaluator()
json_schema_evaluator = JsonSchemaEvaluator()

results = []
for doc_name, articles_ref, doc_content in iterate_benchmark_files(
    Path("./data/documents/")
):
    model_lab = ModelLaboratory.from_llms(llms, prompt=prompt)
    for i, chain in enumerate(model_lab.chains):
        result = {
            "model": get_model_name(llms[i]),
            "doc_name": doc_name,
        }
        try:
            logger.info(f"Running {get_model_name(llms[i])} on document {doc_name}")
            start_time = time.time()
            model_ouput = output_parser.parse(chain.run(doc_content))
            logger.info(model_ouput)
            end_time = time.time()
            prediction = normalize_output(model_ouput.json())
            result["dl_distance"] = json_distance_evaluator.evaluate_strings(
                prediction=prediction,
                reference=normalize_output(articles_ref.json()),
            )["score"]
            result["valid_schema"] = json_schema_evaluator.evaluate_strings(
                prediction=prediction,
                reference=DocumentLawArticles.schema_json(),
            )["score"]
            result["duration"] = end_time - start_time
            result["model_output"] = prediction

        except Exception as e:
            # Log des modèles qui ont échoué. On continue le benchmark tout en conservant l'erreur dans le résultat.
            logger.error(
                f"Error while running {get_model_name(llms[i])} on document {doc_name}"
            )
            logger.error(e)
            result["valid_schema"] = False
            result["model_output"] = str(e)
        finally:
            results.append(result)

# %% Sauvegarde des résultats
logger.info(results)
df = pd.DataFrame.from_dict(results)
df.to_csv(f"results-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv")
