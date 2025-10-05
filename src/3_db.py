# db.py
# - Conecta via DATABASE_URL (ou variáveis separadas) com sslmode=require por padrão
# - Usa pool de conexões (psycopg2.pool.SimpleConnectionPool)
# - Context managers para conexão e cursor (commit/rollback automáticos)
# - UPSERT do documento só é aplicado se houver UNIQUE(filename); caso contrário, mantém INSERT
# - Inserção em lote (chunks) com execute_values + RETURNING id
# - Transação única para documento + chunks
# - Suporte opcional a pgvector (se pacote "pgvector" estiver instalado, registra adaptador)
#
# Observação importante:
#   Este código NÃO altera seu schema do Supabase. Se a coluna documents.filename
#   não tiver UNIQUE, os métodos "upsert" farão INSERT (comportamento atual).
#
# Requisitos:
#   pip install psycopg2-binary python-dotenv (opcional em dev)
#   pip install pgvector (opcional, apenas se suas colunas embedding forem do tipo "vector")

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, List, Sequence, Tuple, Dict, Any

import psycopg2
from psycopg2 import pool, extras

# Carrega .env apenas se disponível (útil em desenvolvimento)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Suporte opcional a pgvector (se instalado)
_PGVECTOR_AVAILABLE = False
try:
    # pip install pgvector
    from pgvector.psycopg2 import register_vector as _register_vector  # type: ignore
    _PGVECTOR_AVAILABLE = True
except Exception:
    _PGVECTOR_AVAILABLE = False
    _register_vector = None  # type: ignore


# =========================
# Configuração e Pool
# =========================

SCHEMA = os.getenv("PGSCHEMA", "public")

def _build_dsn() -> str:
    """
    Retorna o DSN para o Postgres. Prioriza DATABASE_URL.
    Se ausente, monta a partir de variáveis separadas, aplicando sslmode=require por padrão.
    """
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn  # Supabase normalmente já inclui sslmode=require
    # Fallback por variáveis separadas (compatível com o "exemplo supabase")
    user = os.getenv("PGUSER") or os.getenv("user")
    password = os.getenv("PGPASSWORD") or os.getenv("password")
    host = os.getenv("PGHOST") or os.getenv("host")
    port = os.getenv("PGPORT") or os.getenv("port") or "5432"
    dbname = os.getenv("PGDATABASE") or os.getenv("dbname")
    sslmode = os.getenv("PGSSLMODE", "require")
    if not all([user, password, host, dbname]):
        raise RuntimeError(
            "Config do banco ausente. Defina DATABASE_URL ou as variáveis "
            "PGUSER/PGPASSWORD/PGHOST/PGDATABASE."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"

_DSN = None  # inicializado lazy
_POOL: Optional[pool.SimpleConnectionPool] = None

def _get_pool() -> pool.SimpleConnectionPool:
    """
    Inicializa o pool sob demanda (lazy). Evita falha na importação caso env não esteja presente.
    """
    global _POOL, _DSN
    if _POOL is None:
        if _DSN is None:
            _DSN = _build_dsn()
        # Ajuste PGMAXCONN via env se quiser
        max_conn = int(os.getenv("PGMAXCONN", "5"))
        _POOL = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=max_conn,
            dsn=_DSN,
            connect_timeout=5,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
            options='-c statement_timeout=15000'  # 15s por statement
        )
    return _POOL

def close_pool() -> None:
    """
    Fecha todas as conexões do pool (use ao encerrar a aplicação).
    """
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None


@contextmanager
def get_conn():
    """
    Context manager para obter/usar uma conexão do pool.
    - Faz commit se sucesso, rollback se exceção.
    - Registra adaptador do pgvector se disponível.
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        # Registro do adaptador pgvector (se instalado).
        if _PGVECTOR_AVAILABLE and _register_vector is not None:
            try:
                _register_vector(conn)
            except Exception:
                # Se falhar por OID ou algo específico, apenas segue.
                pass
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = False):
    """
    Cursor com opcional RealDictCursor.
    """
    with get_conn() as conn:
        factory = extras.RealDictCursor if dict_cursor else None
        with conn.cursor(cursor_factory=factory) as cur:
            yield cur


# =========================
# Introspecção de Schema
# =========================

# Cache de informações do schema para evitar consultas repetidas
_SCHEMA_INFO: Optional[Dict[str, Any]] = None

def _get_embedding_type(table: str) -> str:
    """
    Retorna o tipo da coluna 'embedding' na tabela informada:
      - 'vector'    -> tipo pgvector
      - 'array'     -> tipo array (ex.: float8[])
      - 'none'      -> coluna não existe
      - 'other'     -> outro tipo
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = 'embedding'
            """,
            (SCHEMA, table)
        )
        row = cur.fetchone()
        if not row:
            return "none"
        data_type, udt_name = row
        # vector é USER-DEFINED + udt_name = 'vector'
        if udt_name == "vector":
            return "vector"
        if data_type == "ARRAY":
            return "array"
        return "other"

def _has_unique_on_exact_columns(table: str, columns: Sequence[str]) -> bool:
    """
    Verifica se existe uma constraint UNIQUE ou PRIMARY KEY exatamente nas colunas informadas.
    Não altera schema. Retorna True/False.
    """
    cols_tuple = tuple(columns)
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
             AND tc.table_name = kcu.table_name
            WHERE tc.table_schema = %s
              AND tc.table_name = %s
              AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
            """,
            (SCHEMA, table)
        )
        constraints = [r[0] for r in cur.fetchall()]

        for cname in constraints:
            cur.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.key_column_usage kcu
                WHERE kcu.constraint_name = %s
                  AND kcu.table_schema = %s
                  AND kcu.table_name = %s
                ORDER BY kcu.ordinal_position
                """,
                (cname, SCHEMA, table)
            )
            cols = tuple(r[0] for r in cur.fetchall())
            if cols == cols_tuple:
                return True
    return False

def _introspect_schema_if_needed() -> Dict[str, Any]:
    global _SCHEMA_INFO
    if _SCHEMA_INFO is None:
        _SCHEMA_INFO = {
            "documents_embedding": _get_embedding_type("documents"),
            "chunks_embedding": _get_embedding_type("chunks"),
            "documents_unique_filename": _has_unique_on_exact_columns("documents", ["filename"]),
        }
    return _SCHEMA_INFO


# =========================
# Helpers de SQL
# =========================

def _embedding_cast_suffix(table: str) -> str:
    """
    Retorna '::vector' se a coluna embedding da tabela for pgvector; caso contrário, string vazia.
    """
    info = _introspect_schema_if_needed()
    t = "documents_embedding" if table == "documents" else "chunks_embedding"
    return "::vector" if info.get(t) == "vector" else ""

def _normalize_vector_param(embedding: Optional[Sequence[float]]) -> Optional[Any]:
    """
    Normaliza o parâmetro de embedding quando o tipo da coluna é pgvector.
    - Se pgvector estiver registrado (pacote instalado), pode passar lista de floats normalmente.
    - Se não estiver, você pode ainda passar lista: algumas instalações funcionarão.
      Como fallback simples, você pode converter para string '[x, y, z]' — porém isso só seria
      necessário se o adaptador não estiver disponível e sua instalação exigir literal.
    Aqui mantemos a lista; a conversão para literal pode ser adicionada se necessário.
    """
    return embedding  # manter padrão; registro do pgvector (se disponível) já ajuda

def _normalize_array_param(embedding: Optional[Sequence[float]]) -> Optional[Any]:
    """
    Para colunas float8[] (ARRAY), enviar lista de floats já funciona nativamente no psycopg2.
    """
    return embedding

def _prepare_embedding_param(table: str, embedding: Optional[Sequence[float]]) -> Optional[Any]:
    info = _introspect_schema_if_needed()
    t = "documents_embedding" if table == "documents" else "chunks_embedding"
    if info.get(t) == "vector":
        return _normalize_vector_param(embedding)
    elif info.get(t) == "array":
        return _normalize_array_param(embedding)
    else:
        # Se não houver coluna de embedding ou for outro tipo, passa o valor como está (geralmente None)
        return embedding


# =========================
# API de Persistência
# =========================

def upsert_document(
    filename: str,
    lang: str,
    doc_type: str,
    content: str,
    source_url: Optional[str] = None,
    embedding: Optional[Sequence[float]] = None
) -> int:
    """
    Insere ou atualiza um documento.
    - Se existir UNIQUE(filename), faz UPSERT (ON CONFLICT).
    - Caso não exista UNIQUE(filename), mantém o comportamento atual (INSERT) para não alterar seu Supabase.
    Retorna o id do documento.
    """
    info = _introspect_schema_if_needed()
    has_unique = info.get("documents_unique_filename", False)
    emb_cast = _embedding_cast_suffix("documents")
    emb_param = _prepare_embedding_param("documents", embedding)

    if has_unique:
        query = f"""
        INSERT INTO {SCHEMA}.documents (source_url, filename, lang, doc_type, content, embedding)
        VALUES (%s, %s, %s, %s, %s, %s{emb_cast})
        ON CONFLICT (filename) DO UPDATE
          SET source_url = EXCLUDED.source_url,
              lang       = EXCLUDED.lang,
              doc_type   = EXCLUDED.doc_type,
              content    = EXCLUDED.content,
              embedding  = EXCLUDED.embedding
        RETURNING id
        """
        params = (source_url, filename, lang, doc_type, content, emb_param)
    else:
        # Sem UNIQUE(filename), apenas INSERT (compatível com seu código atual)
        query = f"""
        INSERT INTO {SCHEMA}.documents (source_url, filename, lang, doc_type, content, embedding)
        VALUES (%s, %s, %s, %s, %s, %s{emb_cast})
        RETURNING id
        """
        params = (source_url, filename, lang, doc_type, content, emb_param)

    with get_cursor() as cur:
        cur.execute(query, params)
        return int(cur.fetchone()[0])


def insert_chunk(
    document_id: int,
    chunk_index: int,
    content: str,
    page_num: Optional[int],
    char_start: Optional[int],
    char_end: Optional[int],
    lang: str,
    headings: Optional[List[str]],
    is_table: bool,
    is_code: bool,
    embedding: Optional[Sequence[float]]
) -> int:
    """
    Insere um único chunk e retorna seu id.
    """
    emb_cast = _embedding_cast_suffix("chunks")
    emb_param = _prepare_embedding_param("chunks", embedding)

    query = f"""
    INSERT INTO {SCHEMA}.chunks
    (document_id, chunk_index, content, page_num, char_start, char_end,
     lang, headings, is_table, is_code, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s{emb_cast})
    RETURNING id
    """
    params = (
        document_id, chunk_index, content, page_num, char_start, char_end,
        lang, headings, is_table, is_code, emb_param
    )
    with get_cursor() as cur:
        cur.execute(query, params)
        return int(cur.fetchone()[0])


def insert_chunks_bulk(
    document_id: int,
    rows: Sequence[
        Tuple[
            int,                      # chunk_index
            str,                      # content
            Optional[int],            # page_num
            Optional[int],            # char_start
            Optional[int],            # char_end
            str,                      # lang
            Optional[List[str]],      # headings
            bool,                     # is_table
            bool,                     # is_code
            Optional[Sequence[float]] # embedding
        ]
    ]
) -> List[int]:
    """
    Insere vários chunks em lote e retorna a lista de ids gerados.
    rows: lista de tuplas conforme a assinatura acima (sem document_id).
    """
    if not rows:
        return []

    emb_cast = _embedding_cast_suffix("chunks")

    # Normaliza os parâmetros (especialmente embedding) e injeta document_id
    values = []
    for r in rows:
        (
            chunk_index, content, page_num, char_start, char_end,
            lang, headings, is_table, is_code, embedding
        ) = r
        emb_param = _prepare_embedding_param("chunks", embedding)
        values.append((
            document_id, chunk_index, content, page_num, char_start, char_end,
            lang, headings, is_table, is_code, emb_param
        ))

    # Template com cast para embedding quando for pgvector
    template = f"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s{emb_cast})"

    sql = f"""
    INSERT INTO {SCHEMA}.chunks
    (document_id, chunk_index, content, page_num, char_start, char_end,
     lang, headings, is_table, is_code, embedding)
    VALUES %s
    RETURNING id
    """

    with get_cursor() as cur:
        extras.execute_values(cur, sql, values, template=template, page_size=1000)
        ids = [int(row[0]) for row in cur.fetchall()]
        return ids


def upsert_document_with_chunks(
    filename: str,
    lang: str,
    doc_type: str,
    content: str,
    chunks: Sequence[
        Tuple[
            int,                      # chunk_index
            str,                      # content
            Optional[int],            # page_num
            Optional[int],            # char_start
            Optional[int],            # char_end
            str,                      # lang
            Optional[List[str]],      # headings
            bool,                     # is_table
            bool,                     # is_code
            Optional[Sequence[float]] # embedding
        ]
    ],
    source_url: Optional[str] = None,
    embedding: Optional[Sequence[float]] = None
) -> Tuple[int, List[int]]:
    """
    Insere (ou atualiza) um documento e, na mesma transação, insere vários chunks (bulk).
    - UPSERT no documento só se UNIQUE(filename) existir; caso contrário, faz INSERT.
    - Retorna (doc_id, [chunk_ids]).
    """
    info = _introspect_schema_if_needed()
    has_unique = info.get("documents_unique_filename", False)
    doc_emb_cast = _embedding_cast_suffix("documents")
    doc_emb_param = _prepare_embedding_param("documents", embedding)
    chunk_emb_cast = _embedding_cast_suffix("chunks")

    insert_doc_sql = f"""
    INSERT INTO {SCHEMA}.documents (source_url, filename, lang, doc_type, content, embedding)
    VALUES (%s, %s, %s, %s, %s, %s{doc_emb_cast})
    """
    if has_unique:
        insert_doc_sql += """
        ON CONFLICT (filename) DO UPDATE
          SET source_url = EXCLUDED.source_url,
              lang       = EXCLUDED.lang,
              doc_type   = EXCLUDED.doc_type,
              content    = EXCLUDED.content,
              embedding  = EXCLUDED.embedding
        """
    insert_doc_sql += " RETURNING id"

    insert_chunks_sql = f"""
    INSERT INTO {SCHEMA}.chunks
    (document_id, chunk_index, content, page_num, char_start, char_end,
     lang, headings, is_table, is_code, embedding)
    VALUES %s
    RETURNING id
    """
    chunk_template = f"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s{chunk_emb_cast})"

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Documento
            cur.execute(
                insert_doc_sql,
                (source_url, filename, lang, doc_type, content, doc_emb_param)
            )
            doc_id = int(cur.fetchone()[0])

            # Chunks (bulk)
            if chunks:
                values = []
                for r in chunks:
                    (
                        chunk_index, content_c, page_num, char_start, char_end,
                        clang, headings, is_table, is_code, emb
                    ) = r
                    emb_param = _prepare_embedding_param("chunks", emb)
                    values.append((
                        doc_id, chunk_index, content_c, page_num, char_start, char_end,
                        clang, headings, is_table, is_code, emb_param
                    ))
                extras.execute_values(cur, insert_chunks_sql, values, template=chunk_template, page_size=1000)
                chunk_ids = [int(row[0]) for row in cur.fetchall()]
            else:
                chunk_ids = []

            return doc_id, chunk_ids