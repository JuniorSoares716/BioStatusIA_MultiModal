"""
Instrumentação de tempo/throughput da LLM (CrewAI + Ollama).

Antes desta correção, o sistema não media em NENHUM lugar o tempo de geração
da LLM nem o throughput de tokens, apesar do artigo (Seção IV-C) afirmar que
"per-agent feature-extraction latency, LLM generation throughput, and
end-to-end report generation time" eram registrados. Só a latência de
inferência do classificador (µs/ms) era medida. Este módulo fecha essa
lacuna, medindo de forma defensiva (nunca lança exceção) ao redor de cada
chamada `crew().kickoff(...)`.
"""
import time


def medir_llm_kickoff(fn):
    """
    Executa `fn()` — uma chamada do tipo `lambda: MinhaCrew().crew().kickoff(inputs=...)`
    — e mede o tempo de parede da geração. Também tenta extrair uso de tokens
    do objeto `CrewOutput` retornado (atributo `token_usage`, quando disponível
    na versão do CrewAI instalada), para calcular throughput (tokens/s).

    Retorna (resultado_do_kickoff, info_timing). Nunca lança: se a chamada em
    si falhar, a exceção original é propagada (quem chama já trata isso hoje);
    só a EXTRAÇÃO de tokens é protegida, pois é informação opcional/best-effort
    que varia entre versões do CrewAI.
    """
    t0 = time.perf_counter()
    resultado = fn()
    tempo_geracao_s = time.perf_counter() - t0

    info: dict = {
        "tempo_geracao_s": round(tempo_geracao_s, 3),
        "tokens_prompt": None,
        "tokens_completion": None,
        "tokens_total": None,
        "throughput_tokens_s": None,
        "fonte_tokens": None,
    }

    uso = getattr(resultado, "token_usage", None)
    if uso is not None:
        try:
            # CrewAI expõe um objeto UsageMetrics (ou dict, dependendo da versão).
            def _campo(obj, nome):
                if isinstance(obj, dict):
                    return obj.get(nome)
                return getattr(obj, nome, None)

            prompt = _campo(uso, "prompt_tokens")
            completion = _campo(uso, "completion_tokens")
            total = _campo(uso, "total_tokens")

            info["tokens_prompt"] = prompt
            info["tokens_completion"] = completion
            info["tokens_total"] = total
            info["fonte_tokens"] = "crewai.token_usage"

            base_throughput = completion or total
            if base_throughput and tempo_geracao_s > 0:
                info["throughput_tokens_s"] = round(base_throughput / tempo_geracao_s, 2)
        except Exception as e:
            info["erro_tokens"] = str(e)

    return resultado, info


class MedidorEndToEnd:
    """Cronômetro simples para o tempo fim-a-fim de uma requisição de análise
    (desde o recebimento do upload até o relatório consolidado estar pronto),
    incluindo tempo de LLM + extração de features + treino do AutoML."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self.marcos: dict[str, float] = {}

    def marcar(self, nome: str) -> None:
        self.marcos[nome] = round(time.perf_counter() - self._t0, 3)

    def total(self) -> float:
        return round(time.perf_counter() - self._t0, 3)

    def resumo(self) -> dict:
        return {"marcos_s": dict(self.marcos), "tempo_total_s": self.total()}
