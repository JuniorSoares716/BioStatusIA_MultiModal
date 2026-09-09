import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "biostatusia.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analises (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT    NOT NULL,
            imagem    TEXT    NOT NULL,
            categoria TEXT    NOT NULL,
            laudo     TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS resultados_pipeline (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora     TEXT    NOT NULL,
            dataset_path  TEXT,
            n_imagens     INTEGER,
            pipeline_json TEXT,
            melhor_modelo TEXT,
            analise_id    INTEGER,
            FOREIGN KEY (analise_id) REFERENCES analises(id)
        )
    """)
    conn.commit()
    _migrar_v2(conn)
    return conn


def _migrar_v2(conn: sqlite3.Connection) -> None:
    """Adiciona colunas v2 e cria tabela de laudos interativos se não existirem."""
    for col in ("familia_sinal TEXT", "sinal_tipo TEXT"):
        try:
            conn.execute(f"ALTER TABLE resultados_pipeline ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS laudos_interativos (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            resultado_id  INTEGER REFERENCES resultados_pipeline(id),
            trecho_inicio REAL,
            trecho_fim    REAL,
            roi_json      TEXT,
            canal         TEXT,
            slice_idx     INTEGER,
            laudo_foco    TEXT,
            data_hora     TEXT NOT NULL
        )
    """)
    conn.commit()


def salvar(imagem: str, categoria: str, laudo: str) -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO analises (data_hora, imagem, categoria, laudo) VALUES (?,?,?,?)",
        (datetime.now().isoformat(), imagem, categoria, laudo),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def listar() -> list[tuple]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, data_hora, imagem, categoria FROM analises ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def salvar_resultado(
    dataset_path: str,
    n_imagens: int,
    pipeline_data: dict,
    melhor_modelo: str,
    analise_id: int,
    familia_sinal: str = "",
    sinal_tipo: str = "",
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO resultados_pipeline
           (data_hora, dataset_path, n_imagens, pipeline_json, melhor_modelo,
            analise_id, familia_sinal, sinal_tipo)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(),
            dataset_path,
            n_imagens,
            json.dumps(pipeline_data),
            melhor_modelo,
            analise_id,
            familia_sinal,
            sinal_tipo,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def buscar_resultado(resultado_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        """SELECT rp.id, rp.data_hora, rp.dataset_path, rp.n_imagens,
                  rp.pipeline_json, rp.melhor_modelo, rp.analise_id,
                  a.imagem, a.categoria, a.laudo,
                  rp.familia_sinal, rp.sinal_tipo
           FROM resultados_pipeline rp
           LEFT JOIN analises a ON a.id = rp.analise_id
           WHERE rp.id = ?""",
        (resultado_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "data_hora": row[1],
        "dataset_path": row[2],
        "n_imagens": row[3],
        "pipeline": json.loads(row[4]) if row[4] else {},
        "melhor_modelo": row[5],
        "analise_id": row[6],
        "imagem": row[7],
        "categoria": row[8],
        "laudo": row[9],
        "familia_sinal": row[10] or "",
        "sinal_tipo": row[11] or "",
    }


def salvar_laudo_interativo(
    resultado_id: int,
    laudo_foco: str,
    trecho_inicio: float | None = None,
    trecho_fim: float | None = None,
    roi_json: str | None = None,
    canal: str | None = None,
    slice_idx: int | None = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO laudos_interativos
           (resultado_id, trecho_inicio, trecho_fim, roi_json, canal, slice_idx,
            laudo_foco, data_hora)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            resultado_id,
            trecho_inicio,
            trecho_fim,
            roi_json,
            canal,
            slice_idx,
            laudo_foco,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def listar_resultados() -> list[tuple]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, data_hora, n_imagens, melhor_modelo FROM resultados_pipeline ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def listar_resultados_completo(limite: int = 30) -> list[dict]:
    """Retorna metadados completos para popular o seletor de amostra na Aba 4."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT rp.id, rp.data_hora, rp.dataset_path, rp.n_imagens,
                  rp.melhor_modelo, rp.familia_sinal, rp.sinal_tipo,
                  a.categoria
           FROM resultados_pipeline rp
           LEFT JOIN analises a ON a.id = rp.analise_id
           ORDER BY rp.id DESC LIMIT ?""",
        (limite,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "data_hora": r[1][:19] if r[1] else "",
            "dataset_path": r[2] or "",
            "n_imagens": r[3] or 0,
            "melhor_modelo": r[4] or "N/A",
            "familia_sinal": r[5] or "",
            "sinal_tipo": r[6] or "",
            "categoria": r[7] or "",
        }
        for r in rows
    ]
