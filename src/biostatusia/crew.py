import os

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .tools.analise_base_tool import FerramentaAnaliseBase
from .tools.extracao_tool import FerramentaExtrairBiomarcadores
from .tools.tabular_tool import FerramentaAnaliseTabular
from .tools.treino_tool import FerramentaTreinarClassificador
from .tools.sinais_temporais_tool import FerramentaExtrairSinalTemporal
from .tools.dicom_tool import FerramentaExtrairDICOM
from .tools.volumetrico_tool import FerramentaExtrairVolume3D


def _llm() -> LLM:
    """LLM local via Ollama — formato obrigatório: 'ollama/<modelo>'."""
    return LLM(
        model=os.getenv("MODEL", "ollama/qwen2.5:3b"),
        base_url=os.getenv("API_BASE", "http://localhost:11434"),
        temperature=0.1,
    )


# ── Crew original: fluxo de imagem ultrassom ─────────────────────────────────

@CrewBase
class BioStatusIACrew:
    """Equipe original (fluxo completo de imagem):
    engenheiro_pdi → analista_tecnico → cientista_dados → radiologista_ia."""

    @agent
    def engenheiro_pdi(self) -> Agent:
        return Agent(
            config=self.agents_config["engenheiro_pdi"],
            tools=[FerramentaAnaliseBase()],
            llm=_llm(), verbose=True, max_iter=5, max_retry_limit=3,
            allow_delegation=False,
        )

    @agent
    def analista_tecnico(self) -> Agent:
        return Agent(
            config=self.agents_config["analista_tecnico"],
            tools=[FerramentaExtrairBiomarcadores()],
            llm=_llm(), verbose=True, max_iter=5, max_retry_limit=3,
            allow_delegation=False,
        )

    @agent
    def cientista_dados(self) -> Agent:
        return Agent(
            config=self.agents_config["cientista_dados"],
            tools=[FerramentaTreinarClassificador()],
            llm=_llm(), verbose=True, max_iter=5, max_retry_limit=3,
            allow_delegation=False,
        )

    @agent
    def radiologista_ia(self) -> Agent:
        return Agent(
            config=self.agents_config["radiologista_ia"],
            llm=_llm(), verbose=True, max_iter=5, max_retry_limit=3,
            allow_delegation=False,
        )

    @task
    def tarefa_analise_base(self) -> Task:
        return Task(config=self.tasks_config["tarefa_analise_base"], agent=self.engenheiro_pdi())

    @task
    def tarefa_extracao(self) -> Task:
        return Task(config=self.tasks_config["tarefa_extracao"], agent=self.analista_tecnico())

    @task
    def tarefa_treinar_classificador(self) -> Task:
        return Task(config=self.tasks_config["tarefa_treinar_classificador"], agent=self.cientista_dados())

    @task
    def tarefa_laudo(self) -> Task:
        return Task(config=self.tasks_config["tarefa_laudo"], agent=self.radiologista_ia())

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.engenheiro_pdi(), self.analista_tecnico(),
                    self.cientista_dados(), self.radiologista_ia()],
            tasks=[self.tarefa_analise_base(), self.tarefa_extracao(),
                   self.tarefa_treinar_classificador(), self.tarefa_laudo()],
            process=Process.sequential, verbose=True,
        )


# ── Crew tabular ─────────────────────────────────────────────────────────────

@CrewBase
class BioStatusIACrewTabular:
    """Equipe de análise tabular (CSV/TXT)."""

    @agent
    def bioestatistico(self) -> Agent:
        return Agent(
            config=self.agents_config["bioestatistico"],
            tools=[FerramentaAnaliseTabular()],
            llm=_llm(), verbose=True, max_iter=2, max_retry_limit=1,
            allow_delegation=False,
        )

    @task
    def tarefa_laudo_tabular(self) -> Task:
        return Task(config=self.tasks_config["tarefa_laudo_tabular"], agent=self.bioestatistico())

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.bioestatistico()],
            tasks=[self.tarefa_laudo_tabular()],
            process=Process.sequential, verbose=True,
        )


# ── Crew F1: Sinais Temporais ────────────────────────────────────────────────

@CrewBase
class BioStatusIACrewSinal:
    """Equipe para sinais temporais fisiológicos (ECG/EEG/EMG/EOG/PPG/etc.)."""

    @agent
    def analista_sinais_fisiologicos(self) -> Agent:
        return Agent(
            config=self.agents_config["analista_sinais_fisiologicos"],
            tools=[FerramentaExtrairSinalTemporal()],
            llm=_llm(), verbose=True, max_iter=6, max_retry_limit=3,
            allow_delegation=False,
        )

    @agent
    def radiologista_ia(self) -> Agent:
        return Agent(
            config=self.agents_config["radiologista_ia"],
            llm=_llm(), verbose=True, max_iter=5, max_retry_limit=3,
            allow_delegation=False,
        )

    @task
    def tarefa_extracao_temporal(self) -> Task:
        return Task(
            config=self.tasks_config["tarefa_extracao_temporal"],
            agent=self.analista_sinais_fisiologicos(),
        )

    @task
    def tarefa_laudo_sinal(self) -> Task:
        return Task(
            config=self.tasks_config["tarefa_laudo_sinal"],
            agent=self.radiologista_ia(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.analista_sinais_fisiologicos(), self.radiologista_ia()],
            tasks=[self.tarefa_extracao_temporal(), self.tarefa_laudo_sinal()],
            process=Process.sequential, verbose=True,
        )


# ── Crew F3+F4: DICOM 2D e Volumes 3D ────────────────────────────────────────

@CrewBase
class BioStatusIACrewImagem3D:
    """Equipe para imagens DICOM 2D (Raio-X, Mamografia) e volumes 3D (TC, RM, PET)."""

    @agent
    def especialista_imagem_medica(self) -> Agent:
        return Agent(
            config=self.agents_config["especialista_imagem_medica"],
            tools=[FerramentaExtrairDICOM(), FerramentaExtrairVolume3D()],
            llm=_llm(), verbose=True, max_iter=6, max_retry_limit=3,
            allow_delegation=False,
        )

    @agent
    def radiologista_ia(self) -> Agent:
        return Agent(
            config=self.agents_config["radiologista_ia"],
            llm=_llm(), verbose=True, max_iter=5, max_retry_limit=3,
            allow_delegation=False,
        )

    @task
    def tarefa_extracao_imagem_medica(self) -> Task:
        return Task(
            config=self.tasks_config["tarefa_extracao_imagem_medica"],
            agent=self.especialista_imagem_medica(),
        )

    @task
    def tarefa_laudo_sinal(self) -> Task:
        return Task(
            config=self.tasks_config["tarefa_laudo_sinal"],
            agent=self.radiologista_ia(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.especialista_imagem_medica(), self.radiologista_ia()],
            tasks=[self.tarefa_extracao_imagem_medica(), self.tarefa_laudo_sinal()],
            process=Process.sequential, verbose=True,
        )


# ── Crew Laudo Interativo ─────────────────────────────────────────────────────

@CrewBase
class BioStatusIACrewInterativo:
    """
    Crew de uso pontual — executa laudo focado no trecho/região selecionado pelo médico.
    Recebe inputs: dados_selecao, tipo_sinal, biomarcadores_trecho.
    """

    @agent
    def radiologista_ia_interativo(self) -> Agent:
        return Agent(
            config=self.agents_config["radiologista_ia_interativo"],
            llm=_llm(), verbose=True, max_iter=4, max_retry_limit=2,
            allow_delegation=False,
        )

    @task
    def tarefa_laudo_interativo(self) -> Task:
        return Task(
            config=self.tasks_config["tarefa_laudo_interativo"],
            agent=self.radiologista_ia_interativo(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.radiologista_ia_interativo()],
            tasks=[self.tarefa_laudo_interativo()],
            process=Process.sequential, verbose=True,
        )
