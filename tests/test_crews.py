"""Tier 3 — montagem das crews e guardrails da constituição (sem LLM/Ollama).

Constrói as crews (sem executar kickoff) e valida a fiação: nº de agentes/tasks,
processo sequencial e a regra de que os agentes de laudo NÃO têm ferramentas.
"""
import pytest

pytest.importorskip("crewai")

from crewai import Process
from biostatusia.crew import (
    BioStatusIACrew, BioStatusIACrewTabular, BioStatusIACrewSinal,
    BioStatusIACrewImagem3D, BioStatusIACrewInterativo,
)


def test_crew_imagem_wiring():
    c = BioStatusIACrew().crew()
    assert len(c.agents) == 4
    assert len(c.tasks) == 4
    assert c.process == Process.sequential


def test_crew_sinal_wiring():
    c = BioStatusIACrewSinal().crew()
    assert len(c.agents) == 2
    assert len(c.tasks) == 2


def test_crew_imagem3d_wiring():
    assert len(BioStatusIACrewImagem3D().crew().agents) == 2


def test_crew_tabular_wiring():
    c = BioStatusIACrewTabular().crew()
    assert len(c.agents) == 1
    assert len(c.tasks) == 1


def test_radiologista_ia_sem_tools():
    # Guardrail: radiologista_ia nunca recebe ferramentas.
    assert not (BioStatusIACrew().radiologista_ia().tools or [])


def test_radiologista_interativo_sem_tools_e_maxiter():
    a = BioStatusIACrewInterativo().radiologista_ia_interativo()
    assert not (a.tools or [])
    assert a.max_iter == 4


def test_agentes_tecnicos_tem_ferramentas():
    # Os agentes de extração/treino DEVEM ter ferramentas.
    assert BioStatusIACrew().engenheiro_pdi().tools
    assert BioStatusIACrewSinal().analista_sinais_fisiologicos().tools
    assert BioStatusIACrewImagem3D().especialista_imagem_medica().tools
