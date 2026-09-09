# BioStatusIA — Spec-Driven Development (SDD)

Este diretório aplica **Spec-Driven Development** ao BioStatusIA. A ideia central do SDD:
a especificação é a fonte da verdade. Primeiro se descreve **o quê** e **por quê** (spec),
depois **como** (plan), depois se quebra em **tarefas executáveis** (tasks) — e só então se
implementa. O código serve à spec, nunca o contrário.

## Fluxo SDD

```
constitution.md   →   spec.md   →   plan.md   →   tasks.md   →   implementação
(princípios)          (o quê/       (como/        (passos          (código)
                       por quê)      técnico)      verificáveis)
```

1. **Constitution** — princípios invioláveis do projeto. Toda spec e todo plano devem
   respeitá-los. Ver `constitution.md`.
2. **Spec** (`spec.md`) — requisitos, histórias de usuário e critérios de aceite de uma
   página/fluxo. Escrita em linguagem de produto, sem detalhe de implementação.
3. **Plan** (`plan.md`) — desenho técnico: rotas, módulos, contratos de dados, decisões
   e riscos. Amarra a spec ao código real do repositório.
4. **Tasks** (`tasks.md`) — decomposição em tarefas verificáveis, cada uma rastreável a um
   critério de aceite da spec. Marca o que já está `Feito` e o que falta.

## Páginas / fluxos especificados

| ID | Página / Fluxo | Rota Flask | Pasta |
|----|----------------|------------|-------|
| 001 | Tela 1 — Upload & Detecção de Modo | `GET /`, `POST /analisar` | `001-tela-upload/` |
| 002 | Tela 2 — Resultados (4 abas) | `GET /resultados/<id>` | `002-tela-resultados/` |
| 003 | Laudo de Amostra (Aba 4 — Seção A) | `POST /laudo_amostra` | `003-laudo-amostra/` |
| 004 | Laudo Populacional (Aba 4 — Seção B) | `GET /laudo_populacional/<id>` | `004-laudo-populacional/` |

APIs de apoio das telas (`GET /api/historico`, `GET /api/exemplos/<id>`) estão
especificadas dentro de 002 e 003, pois só existem para servir essas páginas.

## Como usar ao evoluir o sistema

Para **qualquer** mudança de página ou fluxo:

1. Atualize primeiro a `spec.md` (o quê muda e por quê) — respeitando a constituição.
2. Atualize a `plan.md` (como implementar).
3. Adicione/atualize itens em `tasks.md` com critério de verificação.
4. Só então altere o código. Ao concluir, marque a task como `Feito`.

Templates reutilizáveis para novas páginas/fluxos estão em `templates/`.

## Skill de referência

A metodologia segue a skill **`tlc-spec-driven`** (Tech Leads Club), instalada via
`npx @tech-leads-club/agent-skills install --force --skill tlc-spec-driven`. Os artefatos
aqui seguem o padrão spec → plan → tasks dessa skill, adaptado ao contexto do BioStatusIA.
